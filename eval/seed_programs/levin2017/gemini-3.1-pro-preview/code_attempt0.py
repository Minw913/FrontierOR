import json
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Solve SAV Routing and Scheduling Optimization")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging intermediate solutions")
    args = parser.parse_args()

    if args.time_limit <= 0:
        return

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    network = instance['network']
    time_params = instance['time_parameters']
    demand = instance['demand']
    fleet = instance['fleet']

    T = time_params['time_horizon_T']
    dt = time_params['time_step_duration_seconds']

    links = []
    link_info = {}
    in_links = {}
    out_links = {}

    centroids = []
    junctions = []

    for n in network['nodes']:
        if n['type'] == 'centroid':
            centroids.append(n['id'])
        else:
            junctions.append(n['id'])

    for l in network['links']:
        u = l['from']
        v = l['to']
        links.append((u, v))
        link_info[(u, v)] = l
        in_links.setdefault(v, []).append((u, v))
        out_links.setdefault(u, []).append((u, v))

    demand_matrix = {}
    for od in demand.get('od_pairs', []):
        c = od['origin']
        s = od['destination']
        for dept_t in od['departure_times']:
            demand_matrix[(c, s, dept_t)] = demand_matrix.get((c, s, dept_t), 0) + 1

    initial_fleet = fleet.get('initial_distribution', {})

    m = gp.Model("SAV_Routing")
    m.setParam('TimeLimit', max(1, args.time_limit - 2))
    m.setParam('Method', 2)  
    m.setParam('Crossover', 0)
    m.setParam('LogToConsole', 0)

    N_U = {}
    N_D = {}
    for (u, v) in links:
        for s in centroids:
            for t in range(T + 1):
                N_U[u, v, s, t] = m.addVar(lb=0.0, name=f"NU_{u}_{v}_{s}_{t}")
                N_D[u, v, s, t] = m.addVar(lb=0.0, name=f"ND_{u}_{v}_{s}_{t}")

    Y = {}
    Y_cent = {}
    for j in junctions:
        for p_l in in_links.get(j, []):
            for next_l in out_links.get(j, []):
                next_l_info = link_info[next_l]
                for s in centroids:
                    if next_l_info['type'] == 'centroid_connector' and next_l[1] in centroids:
                        if s != next_l[1]:
                            continue
                    for t in range(T):
                        Y[p_l, next_l, s, t] = m.addVar(lb=0.0, name=f"Y_{p_l[0]}_{p_l[1]}_{next_l[1]}_{s}_{t}")

    for c in centroids:
        for oc in out_links.get(c, []):
            for s in centroids:
                for t in range(T):
                    Y_cent[c, oc[1], s, t] = m.addVar(lb=0.0, name=f"Yc_{c}_{oc[1]}_{s}_{t}")

    p = {}
    for c in centroids:
        for t in range(T + 1):
            p[c, t] = m.addVar(lb=0.0, name=f"p_{c}_{t}")

    e = {}
    omega = {}
    for c in centroids:
        for s in centroids:
            for t in range(T):
                e[c, s, t] = m.addVar(lb=0.0, name=f"e_{c}_{s}_{t}")
            for t in range(T + 1):
                omega[c, s, t] = m.addVar(lb=0.0, name=f"omega_{c}_{s}_{t}")

    # Initial conditions
    for (u, v) in links:
        for s in centroids:
            m.addConstr(N_U[u, v, s, 0] == 0)
            m.addConstr(N_D[u, v, s, 0] == 0)

    for c in centroids:
        m.addConstr(p[c, 0] == initial_fleet.get(c, 0))
        for s in centroids:
            m.addConstr(omega[c, s, 0] == 0)

    # Dynamics 
    for t in range(T):
        for (u, v) in links:
            l_info = link_info[(u, v)]
            is_road = (l_info['type'] == 'road')
            is_L_in = (l_info['type'] == 'centroid_connector' and v in centroids)
            is_L_out = (l_info['type'] == 'centroid_connector' and u in centroids)

            for s in centroids:
                # Upstream Update
                if is_road or is_L_in:
                    preds = in_links.get(u, [])
                    sum_y_in = gp.quicksum(Y[pred_l, (u, v), s, t] for pred_l in preds if (pred_l, (u, v), s, t) in Y)
                    m.addConstr(N_U[u, v, s, t+1] == N_U[u, v, s, t] + sum_y_in)
                elif is_L_out:
                    m.addConstr(N_U[u, v, s, t+1] == N_U[u, v, s, t] + Y_cent[u, v, s, t])

                # Downstream Update
                if is_road or is_L_out:
                    succs = out_links.get(v, [])
                    sum_y_out = gp.quicksum(Y[(u, v), succ_l, s, t] for succ_l in succs if ((u, v), succ_l, s, t) in Y)
                    m.addConstr(N_D[u, v, s, t+1] == N_D[u, v, s, t] + sum_y_out)
                elif is_L_in:
                    m.addConstr(N_D[u, v, s, t+1] == N_U[u, v, s, t])

        # Link sending and receiving bounds
        for (u, v) in links:
            l_info = link_info[(u, v)]
            is_road = (l_info['type'] == 'road')
            is_L_out = (l_info['type'] == 'centroid_connector' and u in centroids)

            if is_road:
                C_l = float(l_info['capacity_vph']) * dt / 3600.0
                K_l = float(l_info['jam_density_vehicles'])
                w_l = int(l_info['congested_travel_time_steps'])

                sum_out = gp.quicksum(Y[(u, v), succ_l, s, t] 
                                      for succ_l in out_links.get(v, [])
                                      for s in centroids if ((u, v), succ_l, s, t) in Y)
                m.addConstr(sum_out <= C_l)

                sum_in = gp.quicksum(Y[pred_l, (u, v), s, t]
                                     for pred_l in in_links.get(u, [])
                                     for s in centroids if (pred_l, (u, v), s, t) in Y)
                m.addConstr(sum_in <= C_l)

                t_wave = max(0, t - w_l + 1)
                wave_recv = gp.quicksum((N_D[u, v, s, t] - N_U[u, v, s, t_wave]) for s in centroids) + K_l
                m.addConstr(sum_in <= wave_recv)

            if is_road or is_L_out:
                f_l = 1 if is_L_out else int(l_info['free_flow_travel_time_steps'])
                for s in centroids:
                    t_send = max(0, t - f_l + 1)
                    sum_out_s = gp.quicksum(Y[(u, v), succ_l, s, t]
                                            for succ_l in out_links.get(v, []) if ((u, v), succ_l, s, t) in Y)
                    m.addConstr(sum_out_s <= N_U[u, v, s, t_send] - N_D[u, v, s, t])
                    if t < f_l:
                        m.addConstr(sum_out_s == 0)

    # Parking logic per centroid and time
    for t in range(T):
        for c in centroids:
            in_connectors = [l for l in in_links.get(c, [])]
            arr = gp.quicksum((N_U[l[0], l[1], c, t] - N_D[l[0], l[1], c, t]) for l in in_connectors)
            out_connectors = [l for l in out_links.get(c, [])]
            dep = gp.quicksum(Y_cent[c, l[1], s, t] for l in out_connectors for s in centroids)
            
            m.addConstr(p[c, t+1] == p[c, t] + arr - dep)
            m.addConstr(dep <= p[c, t])
            
            for s in centroids:
                d_val = demand_matrix.get((c, s, t), 0)
                m.addConstr(omega[c, s, t+1] == omega[c, s, t] + d_val - e[c, s, t])
                m.addConstr(e[c, s, t] <= omega[c, s, t] + d_val)
                
                dep_to_s = gp.quicksum(Y_cent[c, l[1], s, t] for l in out_connectors)
                m.addConstr(e[c, s, t] <= dep_to_s)

    # Ensure $N_U >= N_D$ strictly due to definitions
    for (u, v) in links:
        for s in centroids:
            for t in range(T+1):
                m.addConstr(N_U[u, v, s, t] >= N_D[u, v, s, t])

    # End state penalties
    slack_omega = {}
    for c in centroids:
        for s in centroids:
            slack_omega[c, s] = m.addVar(lb=0.0)
            m.addConstr(omega[c, s, T] <= slack_omega[c, s])

    sum_p_T = gp.quicksum(p[c, T] for c in centroids)
    sum_p_0 = gp.quicksum(p[c, 0] for c in centroids)
    slack_fleet = m.addVar(lb=0.0)
    m.addConstr(sum_p_0 - sum_p_T <= slack_fleet)
    m.addConstr(sum_p_T - sum_p_0 <= slack_fleet)

    obj_veh = gp.quicksum((N_U[u, v, s, t] - N_D[u, v, s, t]) for (u, v) in links for s in centroids for t in range(T+1))
    obj_wait = gp.quicksum(omega[c, s, t] for c in centroids for s in centroids for t in range(T+1))
    
    m.setObjective(obj_veh + obj_wait + 1e6 * gp.quicksum(slack_omega[c, s] for c in centroids for s in centroids) + 1e6 * slack_fleet, GRB.MINIMIZE)
    m.optimize()

    # Solution logging and serialization
    if m.SolCount > 0:
        base_obj_val = float(obj_veh.getValue() + obj_wait.getValue())
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
            if logger:
                logger.log(base_obj_val)
        except ImportError:
            pass

        ans = {
            "objective_value": base_obj_val,
            "y": {}, "y_centroid": {}, "N_U": {}, "N_D": {}, "p": {}, "e": {}, "omega": {}
        }
        
        for ((u, v), next_l, s, t), var in Y.items():
            ans["y"][f"{u}|{v}|{next_l[1]}|{s}|{t}"] = max(0.0, var.X)
            
        for (c, j, s, t), var in Y_cent.items():
            ans["y_centroid"][f"{c}|{j}|{s}|{t}"] = max(0.0, var.X)
            
        for (u, v, s, t), var in N_U.items():
            ans["N_U"][f"{u}|{v}|{s}|{t}"] = max(0.0, var.X)
            
        for (u, v, s, t), var in N_D.items():
            ans["N_D"][f"{u}|{v}|{s}|{t}"] = max(0.0, var.X)
            
        for (c, t), var in p.items():
            ans["p"][f"{c}|{t}"] = max(0.0, var.X)
            
        for (c, s, t), var in e.items():
            ans["e"][f"{c}|{s}|{t}"] = max(0.0, var.X)
            
        for (c, s, t), var in omega.items():
            ans["omega"][f"{c}|{s}|{t}"] = max(0.0, var.X)

        with open(args.solution_path, 'w') as f:
            json.dump(ans, f, indent=2)

if __name__ == "__main__":
    main()