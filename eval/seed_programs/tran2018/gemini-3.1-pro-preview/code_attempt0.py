import argparse
import json
import logging
import math
import bisect
import gurobipy as gp
from gurobipy import GRB
import sys

def main():
    parser = argparse.ArgumentParser(description="Optimal Refueling Station Location")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the generic input JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum execution time limit in seconds.")
    parser.add_argument("--log_path", type=str, required=False, help="Path to save progress logs.", default=None)
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['parameters']['n']
    p_req = data['parameters']['p']
    R = data['parameters']['R']
    R_tol = R + 1e-7

    # Validate logical bound for number of stations based on unique locations
    p = min(n, p_req)

    nodes_dict = {node['id']: node for node in data['nodes']}

    # Store edge distances using standardized unordered pairs
    edge_dist = {}
    for arc in data['arcs']:
        u = arc['from']
        v = arc['to']
        d = arc['distance']
        if u > v:
            u, v = v, u
        edge_dist[(u, v)] = d

    OD_fulfilled_free = []
    OD_impossible = []
    OD_conditions = {}

    for q_idx, od in enumerate(data['od_pairs']):
        path = od['shortest_path']
        if len(path) <= 1:
            OD_fulfilled_free.append(q_idx)
            continue
        
        # Unroll the forward and return route
        w = path + path[-2::-1]
        m_len = len(w)
        
        # Sequentially map cumulative distances
        D = [0.0] * m_len
        for i in range(1, m_len):
            u, v = w[i-1], w[i]
            if u > v:
                u, v = v, u
            
            if (u, v) in edge_dist:
                d = edge_dist[(u, v)]
            else:
                n1, n2 = nodes_dict.get(u), nodes_dict.get(v)
                if n1 and n2:
                    d = math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])
                else:
                    d = 0.0
            
            D[i] = D[i-1] + d

        # Check if the entire roundtrip falls within vehicle range
        if D[-1] <= R_tol:
            OD_fulfilled_free.append(q_idx)
            continue
            
        impossible = False
        C_list = []
        for i in range(1, m_len):
            if D[i] > R_tol:
                min_D = D[i] - R_tol
                # Binary search for efficiently finding the earliest capable originating refueling spot index
                j_start = bisect.bisect_left(D, min_D, 1, i)
                if j_start == i:
                    impossible = True
                    break
                # Deduplicate physical node occurrences inside this workable slice
                C_i = set(w[j_start:i])
                C_list.append(C_i)
                
        if impossible:
            OD_impossible.append(q_idx)
        else:
            # Minimal viable set refinement natively handles pruning logically inferior redundant node setups
            C_list.sort(key=len)
            min_sets = []
            for C in C_list:
                if not any(M.issubset(C) for M in min_sets):
                    min_sets.append(C)
            OD_conditions[q_idx] = min_sets

    # Set up and formulate Mixed Integer Program
    model = gp.Model("StationLocation")
    model.setParam('TimeLimit', float(args.time_limit))
    model.setParam('Threads', 1)

    y = {}
    node_ids = list(nodes_dict.keys())
    for nid in node_ids:
        y[nid] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}")

    x = {}
    for q in OD_conditions:
        x[q] = model.addVar(vtype=GRB.BINARY, name=f"x_{q}")

    model.addConstr(gp.quicksum(y.values()) == p, name="p_stations")

    for q, min_sets in OD_conditions.items():
        for idx, S in enumerate(min_sets):
            model.addConstr(gp.quicksum(y[v] for v in S) >= x[q], name=f"cov_{q}_{idx}")

    # Build objective expression efficiently and safely tracking constants
    sum_free_flow = sum(data['od_pairs'][q]['flow'] for q in OD_fulfilled_free)
    obj_expr = gp.LinExpr()
    obj_expr.addConstant(sum_free_flow)
    for q in OD_conditions:
        obj_expr.addTerms(data['od_pairs'][q]['flow'], x[q])
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Supply an initial feasible scenario mathematically guaranteeing bounded fallback
    for i, nid in enumerate(node_ids):
        y[nid].Start = 1 if i < p else 0
    for q in OD_conditions:
        x[q].Start = 0

    # Conditionally integrate SolutionLogger framework as requested target hook
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except ImportError:
            pass

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(cb)

    # Fallback bounds enforcement mechanism ensuring non-fault compliance
    if model.SolCount > 0:
        selected_stations = [nid for nid in node_ids if y[nid].X > 0.5]
    else:
        selected_stations = node_ids[:p]

    if len(selected_stations) > p:
        selected_stations = selected_stations[:p]
    elif len(selected_stations) < p:
        for nid in node_ids:
            if nid not in selected_stations:
                selected_stations.append(nid)
            if len(selected_stations) == p: break

    # Accurately and strictly recount optimally met origin-destination sets based on the ultimate deployed layout
    selected_set = set(selected_stations)
    refueled_od_pairs = list(OD_fulfilled_free)

    for q, min_sets in OD_conditions.items():
        fulfilled = True
        for S in min_sets:
            if not (S & selected_set):
                fulfilled = False
                break
        if fulfilled:
            refueled_od_pairs.append(q)

    refueled_od_pairs.sort()
    final_obj = float(sum(data['od_pairs'][q]['flow'] for q in refueled_od_pairs))

    # Concat formatting ensuring JSON schema conformity
    sol = {
        "objective_value": final_obj,
        "stations": selected_stations,
        "refueled_od_pairs": refueled_od_pairs
    }

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

if __name__ == '__main__':
    main()