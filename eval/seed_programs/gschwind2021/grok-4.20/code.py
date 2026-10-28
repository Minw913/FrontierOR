import json
import argparse
import time
import sys
from typing import List, Dict, Any
import networkx as nx
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

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

    graph_data = instance['graph']
    settings = instance['problem_settings']

    G = nx.Graph()
    G.add_nodes_from(graph_data['vertices'])
    G.add_edges_from(graph_data['edges'])

    n = graph_data['num_vertices']
    vertices = sorted(graph_data['vertices'])
    v_map = {v: i for i, v in enumerate(vertices)}

    relaxed_type = settings['relaxed_clique_type']
    param = settings.get('parameter_s')
    if param is None and 'parameter_gamma' in settings:
        param = settings['parameter_gamma']
    decomp_type = settings['decomposition_type']
    connectivity = settings['connectivity_required']

    start_time = time.time()
    time_limit = args.time_limit

    # Generate candidate groups - use maximal cliques + some heuristic groups for relaxation
    candidates = []
    # Always include singletons
    for v in vertices:
        candidates.append([v])

    # Find maximal cliques as base
    try:
        cliques = list(nx.find_cliques(G))
        for c in cliques:
            if len(c) > 1:
                candidates.append(sorted(c))
    except:
        pass

    # For relaxed types, add some additional groups by expanding cliques or random sampling
    if relaxed_type in ['s-plex', 's-clique', 's-club', 's-defective', 's-bundle', 'gamma-quasi-clique']:
        s = int(param) if isinstance(param, (int, float)) and relaxed_type != 'gamma-quasi-clique' else 1
        # Add neighborhoods
        for v in vertices:
            neigh = list(G.neighbors(v))
            if len(neigh) >= 2:
                group = sorted([v] + neigh[:min(5, len(neigh))])
                candidates.append(group)
        
        # Add some larger connected components with relaxation allowance
        for seed in vertices:
            group = [seed]
            for _ in range(min(6, n-1)):
                candidates_neigh = set()
                for u in group:
                    for nei in G.neighbors(u):
                        if nei not in group:
                            candidates_neigh.add(nei)
                if not candidates_neigh:
                    break
                group.append(list(candidates_neigh)[0])
            if len(group) > 1:
                candidates.append(sorted(group))

    # Remove duplicates
    unique_cands = []
    seen = set()
    for c in candidates:
        ct = tuple(c)
        if ct not in seen:
            seen.add(ct)
            unique_cands.append(c)
    candidates = unique_cands

    # Filter candidates based on relaxed clique definition and connectivity
    valid_groups = []
    for group in candidates:
        if len(group) == 1:
            valid_groups.append(group)
            continue
        if connectivity:
            induced = G.subgraph(group)
            if not nx.is_connected(induced):
                continue
        if is_valid_relaxed_clique(G, group, relaxed_type, param, connectivity):
            valid_groups.append(sorted(group))

    # If too many candidates, limit to reduce model size (heuristic)
    if len(valid_groups) > 800:
        # Keep all singletons and larger groups preferentially
        singletons = [g for g in valid_groups if len(g) == 1]
        others = [g for g in valid_groups if len(g) > 1]
        others.sort(key=len, reverse=True)
        valid_groups = singletons + others[:700]

    m = len(valid_groups)
    if m == 0:
        # Fallback: just use singletons
        valid_groups = [[v] for v in vertices]
        m = n

    # Create model
    model = gp.Model("RelaxedCliqueDecomp")
    model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)  # Allow some gap for larger instances
    model.setParam('LogToConsole', 0)

    # Variables: x_j = 1 if group j is selected
    x = model.addVars(m, vtype=GRB.BINARY, name="x")

    # Objective: minimize number of groups
    model.setObjective(gp.quicksum(x[j] for j in range(m)), GRB.MINIMIZE)

    # Coverage constraints
    vertex_to_groups = [[] for _ in range(n)]
    for j, group in enumerate(valid_groups):
        for v in group:
            idx = v_map[v]
            vertex_to_groups[idx].append(j)

    for i in range(n):
        if decomp_type == 'partitioning':
            model.addConstr(gp.quicksum(x[j] for j in vertex_to_groups[i]) == 1, f"part_{i}")
        else:  # covering
            model.addConstr(gp.quicksum(x[j] for j in vertex_to_groups[i]) >= 1, f"cover_{i}")

    # Callback for logging incumbents
    incumbent_obj = float('inf')
    def callback(model, where):
        nonlocal incumbent_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < incumbent_obj - 1e-6:
                incumbent_obj = obj
                if logger:
                    logger.log(float(obj))

    # Solve
    model.optimize(callback)

    # Extract solution
    selected = []
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        for j in range(m):
            if x[j].X > 0.5:
                selected.append(valid_groups[j])
        objective = len(selected)
    else:
        # Fallback: use all singletons
        selected = [[v] for v in vertices]
        objective = n

    # Final logging
    if logger and objective < incumbent_obj:
        logger.log(float(objective))

    # Prepare output
    output = {
        "objective_value": float(objective),
        "solution": selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

def is_valid_relaxed_clique(G: nx.Graph, group: List[int], rtype: str, param: Any, connectivity: bool) -> bool:
    if len(group) == 1:
        return True
    induced = G.subgraph(group)
    n_g = len(group)
    if connectivity and not nx.is_connected(induced):
        return False
    try:
        if rtype == 's-plex':
            s = int(param)
            min_degree = n_g - s
            degrees = dict(induced.degree())
            return all(d >= min_degree for d in degrees.values())
        elif rtype == 'k-core':
            k = int(param)
            degrees = dict(induced.degree())
            return all(d >= k for d in degrees.values())
        elif rtype == 's-clique':
            s = int(param)
            # Distance in original graph
            all_pairs = dict(nx.all_pairs_shortest_path_length(G.subgraph(group)))
            for u in group:
                for v in group:
                    if u != v and all_pairs[u].get(v, 999) > s:
                        return False
            return True
        elif rtype == 's-club':
            s = int(param)
            # Diameter of induced
            if n_g <= 1:
                return True
            try:
                diam = nx.diameter(induced)
                return diam <= s
            except:
                return False
        elif rtype == 'gamma-quasi-clique':
            gamma = float(param)
            e = induced.number_of_edges()
            max_e = n_g * (n_g - 1) // 2
            if max_e == 0:
                return True
            return (e / max_e) >= gamma
        elif rtype == 's-defective':
            s = int(param)
            e = induced.number_of_edges()
            max_e = n_g * (n_g - 1) // 2
            return e >= max_e - s
        elif rtype == 'k-block':
            k = int(param)
            if n_g <= k:
                return False
            try:
                conn = nx.node_connectivity(induced)
                return conn >= k
            except:
                return False
        elif rtype == 's-bundle':
            s = int(param)
            if n_g <= s:
                return False
            min_conn = n_g - s
            try:
                conn = nx.node_connectivity(induced)
                return conn >= min_conn
            except:
                return False
        else:
            # Default to clique if unknown type
            return nx.is_clique(G, group)
    except:
        return False

if __name__ == "__main__":
    main()