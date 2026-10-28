import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from collections import defaultdict

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

    network = instance['network']
    nodes = {n['id']: n for n in network['nodes']}
    links = network['links']
    link_map = {l['id']: l for l in links}
    
    T = instance['time_parameters']['time_horizon_T']
    time_step_dur = instance['time_parameters']['time_step_duration_seconds']
    
    # Determine destinations
    D = set()
    for od in instance['demand']['od_pairs']:
        D.add(od['destination'])
    D = sorted(list(D))
    
    # Precompute links by type
    L_road = []
    L_out = []
    L_in = []
    for l in links:
        if l['type'] == 'road':
            L_road.append(l)
        elif l['type'] == 'centroid_connector':
            if nodes[l['from']]['type'] == 'centroid':
                L_out.append(l)
            else:
                L_in.append(l)
                
    # Successors and predecessors
    succ = defaultdict(list)
    pred = defaultdict(list)
    for l in L_road:
        for l2 in L_road:
            if l['to'] == l2['from']:
                succ[l['id']].append(l2['id'])
                pred[l2['id']].append(l['id'])
        for l2 in L_in:
            if l['to'] == l2['from']:
                succ[l['id']].append(l2['id'])
                pred[l2['id']].append(l['id'])
    for l in L_out:
        for l2 in L_road:
            if l['to'] == l2['from']:
                succ[l['id']].append(l2['id'])
                pred[l2['id']].append(l['id'])
        for l2 in L_in:
            if l['to'] == l2['from']:
                succ[l['id']].append(l2['id'])
                pred[l2['id']].append(l['id'])
                
    L_in_by_centroid = defaultdict(list)
    L_out_by_centroid = defaultdict(list)
    for l in L_in:
        L_in_by_centroid[l['to']].append(l['id'])
    for l in L_out:
        L_out_by_centroid[l['from']].append(l['id'])
        
    # Demand aggregation
    demand = defaultdict(lambda: defaultdict(float))
    for od in instance['demand']['od_pairs']:
        o = od['origin']
        d = od['destination']
        for t in od['departure_times']:
            demand[o][d][t] += 1.0

    model = gp.Model("SAV_LTM")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Variables
    N_U = {}
    N_D = {}
    y = {}
    y_centroid = {}
    p = {}
    e = {}
    omega = {}
    
    for d in D:
        for l in L_road + L_out:
            for t in range(T + 1):
                N_U[(l['id'], d, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"NU_{l['id']}_{d}_{t}")
                N_D[(l['id'], d, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ND_{l['id']}_{d}_{t}")
        for l in L_in:
            if l['to'] == d:
                for t in range(T + 1):
                    N_U[(l['id'], d, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"NU_{l['id']}_{d}_{t}")
                    N_D[(l['id'], d, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ND_{l['id']}_{d}_{t}")

    for d in D:
        for l in L_road + L_out:
            for t in range(T):
                for l2_id in succ[l['id']]:
                    l2 = link_map[l2_id]
                    y[(l['id'], l2_id, d, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"y_{l['id']}_{l2_id}_{d}_{t}")
                    
    for c in nodes:
        if nodes[c]['type'] == 'centroid':
            for d in D:
                for t in range(T + 1):
                    p[(c, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"p_{c}_{t}")
                    omega[(c, d, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"omega_{c}_{d}_{t}")
            for d in D:
                for t in range(T):
                    e[(c, d, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"e_{c}_{d}_{t}")
            for j in [l['to'] for l in L_out if l['from'] == c]:
                for d in D:
                    for t in range(T):
                        y_centroid[(c, j, d, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"y_c_{c}_{j}_{d}_{t}")
                        
    # Constraints
    for d in D:
        for l in L_road + L_out:
            model.addConstr(N_U[(l['id'], d, 0)] == 0)
            model.addConstr(N_D[(l['id'], d, 0)] == 0)
        for l in L_in:
            if l['to'] == d:
                model.addConstr(N_U[(l['id'], d, 0)] == 0)
                model.addConstr(N_D[(l['id'], d, 0)] == 0)
                
    for c in nodes:
        if nodes[c]['type'] == 'centroid':
            model.addConstr(p[(c, 0)] == instance['fleet']['initial_distribution'].get(c, 0.0))
            for d in D:
                model.addConstr(omega[(c, d, 0)] == 0)

    for d in D:
        for t in range(T):
            # N_U and N_D updates
            for l in L_out:
                c = l['from']
                j = l['to']
                model.addConstr(N_U[(l['id'], d, t+1)] == N_U[(l['id'], d, t)] + y_centroid[(c, j, d, t)])
                out_flow = gp.quicksum(y[(l['id'], l2_id, d, t)] for l2_id in succ[l['id']])
                model.addConstr(N_D[(l['id'], d, t+1)] == N_D[(l['id'], d, t)] + out_flow)
                
            for l in L_road:
                out_flow = gp.quicksum(y[(l['id'], l2_id, d, t)] for l2_id in succ[l['id']])
                model.addConstr(N_U[(l['id'], d, t+1)] == N_U[(l['id'], d, t)] + out_flow)
                in_flow = gp.quicksum(y[(l2_id, l['id'], d, t)] for l2_id in pred[l['id']])
                model.addConstr(N_D[(l['id'], d, t+1)] == N_D[(l['id'], d, t)] + in_flow)
                
            for l in L_in:
                if l['to'] == d:
                    in_flow = gp.quicksum(y[(l2_id, l['id'], d, t)] for l2_id in pred[l['id']])
                    model.addConstr(N_U[(l['id'], d, t+1)] == N_U[(l['id'], d, t)] + in_flow)
                    model.addConstr(N_D[(l['id'], d, t+1)] == N_U[(l['id'], d, t)])

    # Sending flow and capacity
    for d in D:
        for t in range(T):
            for l in L_road + L_out:
                tau = l['free_flow_travel_time_steps']
                idx = t - tau + 1
                if idx < 0:
                    S = - N_D[(l['id'], d, t)]
                else:
                    S = N_U[(l['id'], d, idx)] - N_D[(l['id'], d, t)]
                out_flow = gp.quicksum(y[(l['id'], l2_id, d, t)] for l2_id in succ[l['id']])
                model.addConstr(out_flow <= S)

    for l in L_road:
        C_l = l['capacity_vph'] * time_step_dur / 3600.0
        gamma = l['congested_travel_time_steps']
        J_l = l['jam_density_vehicles']
        for t in range(T):
            out_flow = gp.quicksum(y[(l['id'], l2_id, d, t)] for d in D for l2_id in succ[l['id']])
            model.addConstr(out_flow <= C_l)
            in_flow = gp.quicksum(y[(l2_id, l['id'], d, t)] for d in D for l2_id in pred[l['id']])
            model.addConstr(in_flow <= C_l)
            
            idx = t - gamma + 1
            R_sum = 0
            for d in D:
                if idx < 0:
                    R_sum += - N_D[(l['id'], d, t)]
                else:
                    R_sum += N_U[(l['id'], d, idx)] - N_D[(l['id'], d, t)]
            R_sum += J_l
            model.addConstr(in_flow <= R_sum)

    # Park constraints
    for c in nodes:
        if nodes[c]['type'] == 'centroid':
            for t in range(T):
                arrivals = 0
                for l_id in L_in_by_centroid[c]:
                    l = link_map[l_id]
                    if l['to'] in D:
                        arrivals += N_U[(l_id, l['to'], t)] - N_D[(l_id, l['to'], t)]
                
                departures = 0
                for l_id in L_out_by_centroid[c]:
                    l = link_map[l_id]
                    j = l['to']
                    for d in D:
                        departures += y_centroid[(c, j, d, t)]
                
                model.addConstr(p[(c, t+1)] == p[(c, t)] + arrivals - departures)
                model.addConstr(departures <= p[(c, t)])
                
            # Final park conservation
            model.addConstr(gp.quicksum(p[(c, T)] for c in nodes if nodes[c]['type'] == 'centroid') == sum(instance['fleet']['initial_distribution'].values()))

    # Pickups
    for c in nodes:
        if nodes[c]['type'] == 'centroid':
            for d in D:
                for t in range(T):
                    model.addConstr(e[(c, d, t)] <= omega[(c, d, t)])
                    dep_d = 0
                    for l_id in L_out_by_centroid[c]:
                        l = link_map[l_id]
                        j = l['to']
                        dep_d += y_centroid[(c, j, d, t)]
                    model.addConstr(e[(c, d, t)] <= dep_d)
                    
                    D_cdt = demand[c][d][t] if t in demand[c][d] else 0.0
                    model.addConstr(omega[(c, d, t+1)] == omega[(c, d, t)] + D_cdt - e[(c, d, t)])
                model.addConstr(omega[(c, d, T)] == 0)

    # Objective
    obj = 0
    for t in range(T):
        for l in L_road + L_out:
            for d in D:
                obj += N_U[(l['id'], d, t)] - N_D[(l['id'], d, t)]
        for l in L_in:
            for d in D:
                if l['to'] == d:
                    obj += N_U[(l['id'], d, t)] - N_D[(l['id'], d, t)]
        for c in nodes:
            if nodes[c]['type'] == 'centroid':
                for d in D:
                    obj += omega[(c, d, t)]
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.optimize()
    
    # Extract solution
    sol = {
        "objective_value": model.objVal if model.SolCount > 0 else 0.0,
        "y": {},
        "y_centroid": {},
        "N_U": {},
        "N_D": {},
        "p": {},
        "e": {},
        "omega": {}
    }
    
    if model.SolCount > 0:
        for d in D:
            for l in L_road + L_out:
                for t in range(T+1):
                    val = N_U[(l['id'], d, t)].X
                    if val > 1e-6:
                        sol["N_U"][f"{l['from']}|{l['to']}|{d}|{t}"] = val
                    val = N_D[(l['id'], d, t)].X
                    if val > 1e-6:
                        sol["N_D"][f"{l['from']}|{l['to']}|{d}|{t}"] = val
            for l in L_in:
                if l['to'] == d:
                    for t in range(T+1):
                        val = N_U[(l['id'], d, t)].X
                        if val > 1e-6:
                            sol["N_U"][f"{l['from']}|{l['to']}|{d}|{t}"] = val
                        val = N_D[(l['id'], d, t)].X
                        if val > 1e-6:
                            sol["N_D"][f"{l['from']}|{l['to']}|{d}|{t}"] = val
                            
        for d in D:
            for l in L_road + L_out:
                for t in range(T):
                    for l2_id in succ[l['id']]:
                        l2 = link_map[l2_id]
                        val = y[(l['id'], l2_id, d, t)].X
                        if val > 1e-6:
                            sol["y"][f"{l['from']}|{l['to']}|{l2['to']}|{d}|{t}"] = val
                            
        for c in nodes:
            if nodes[c]['type'] == 'centroid':
                for t in range(T+1):
                    val = p[(c, t)].X
                    if val > 1e-6:
                        sol["p"][f"{c}|{t}"] = val
                for d in D:
                    for t in range(T+1):
                        val = omega[(c, d, t)].X
                        if val > 1e-6:
                            sol["omega"][f"{c}|{d}|{t}"] = val
                    for t in range(T):
                        val = e[(c, d, t)].X
                        if val > 1e-6:
                            sol["e"][f"{c}|{d}|{t}"] = val
                        for l_id in L_out_by_centroid[c]:
                            l = link_map[l_id]
                            j = l['to']
                            val = y_centroid[(c, j, d, t)].X
                            if val > 1e-6:
                                sol["y_centroid"][f"{c}|{j}|{d}|{t}"] = val
                                
    if logger:
        logger.log_solution(sol["objective_value"], sol)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()