import argparse
import json
import time
import random

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

def solve_with_gurobi(num_vertices, edges, num_clusters, clusters, best_sol, logger, time_left):
    best_selected, best_coloring, U = best_sol
    
    if U <= 1:
        return best_sol

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("PerfectGraphColoring", env=env)
    m.setParam("Threads", 1)
    m.setParam('TimeLimit', time_left)
    
    num_edge_constraints = len(edges) * U
    use_lazy = num_edge_constraints > 2_000_000
    
    if use_lazy:
        m.setParam('LazyConstraints', 1)

    w = m.addVars(U, vtype=GRB.BINARY, name="w")
    y = m.addVars(num_vertices, U, vtype=GRB.BINARY, name="y")

    m.setObjective(gp.quicksum(w[c] for c in range(U)), GRB.MINIMIZE)

    # Symmetry breaking for colors
    for c in range(U - 1):
        m.addConstr(w[c] >= w[c+1])

    # Each cluster gets exactly one selected vertex that has EXACTLY one color assigned
    for i, cluster in enumerate(clusters):
        m.addConstr(gp.quicksum(y[v, c] for v in cluster for c in range(U)) == 1)

    # A vertex can't have a color if that color is not used globally
    for v in range(num_vertices):
        m.addConstr(gp.quicksum(y[v, c] for c in range(U)) <= 1)
        for c in range(U):
            m.addConstr(y[v, c] <= w[c])

    # Add edge constraints directly if not too many
    if not use_lazy:
        m.addConstrs((y[u, c] + y[v, c] <= w[c] for u, v in edges for c in range(U)))

    # Set warmup start from heuristic
    for c in range(U):
        w[c].start = 1
    for v in range(num_vertices):
        for c in range(U):
            y[v, c].start = 0
    for v, color in best_coloring.items():
        y[v, color - 1].start = 1

    edges_set = set()
    if use_lazy:
        for u, v in edges:
            if u > v:
                u, v = v, u
            edges_set.add((u, v))

    added_lazy_edges = set()
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
            
            if use_lazy:
                y_val = model.cbGetSolution(y)
                selected_by_color = {c: [] for c in range(U)}
                for v in range(num_vertices):
                    for c in range(U):
                        if y_val[v, c] > 0.5:
                            selected_by_color[c].append(v)
                
                # Check for conflicts safely and dynamically lazily add
                for c in range(U):
                    nodes = selected_by_color[c]
                    n_nodes = len(nodes)
                    for i in range(n_nodes):
                        for j in range(i+1, n_nodes):
                            u, v = nodes[i], nodes[j]
                            if u > v:
                                u, v = v, u
                            if (u, v) in edges_set and (u, v) not in added_lazy_edges:
                                added_lazy_edges.add((u, v))
                                # Exclude them symmetrically across all colors
                                for all_c in range(U):
                                    model.cbLazy(y[u, all_c] + y[v, all_c] <= w[all_c])

    m.optimize(callback)

    if m.SolCount > 0:
        extracted_selected = {}
        extracted_coloring = {}
        for i, cluster in enumerate(clusters):
            for v in cluster:
                for c in range(U):
                    if y[v, c].X > 0.5:
                        extracted_selected[i] = v
                        extracted_coloring[v] = c + 1
        
        c_map = {}
        curr_k = 1
        for v in extracted_selected.values():
            orig_c = extracted_coloring[v]
            if orig_c not in c_map:
                c_map[orig_c] = curr_k
                curr_k += 1
            extracted_coloring[v] = c_map[orig_c]
            
        extracted_num = curr_k - 1
        if extracted_num < U:
            return extracted_selected, extracted_coloring, extracted_num

    return best_selected, best_coloring, U

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    num_vertices = data['graph']['num_vertices']
    edges_list = data['graph']['edges']
    num_clusters = data['partition']['num_clusters']
    clusters = data['partition']['clusters']
    
    # Establish generic adjacency list mappings
    adj = {i: set() for i in range(num_vertices)}
    for u, v in edges_list:
        adj[u].add(v)
        adj[v].add(u)
        
    global_best_colors = float('inf')
    global_best_selected = {}
    global_best_coloring = {}

    def evaluate(sel_dict):
        sel_nodes_list = list(sel_dict.values())
        m_len = len(sel_nodes_list)
        
        # Precompute induced sub-graph degrees structure optimally O(m^2)
        degrees = {n: 0 for n in sel_nodes_list}
        for i in range(m_len):
            n1 = sel_nodes_list[i]
            for j in range(i+1, m_len):
                n2 = sel_nodes_list[j]
                if n2 in adj[n1]:
                    degrees[n1] += 1
                    degrees[n2] += 1
                    
        sat_colors = {n: set() for n in sel_nodes_list}
        uncolored = set(sel_nodes_list)
        color_assignment = {}
        
        # Optimized DSatur application
        while uncolored:
            best_node = None
            best_sat = -1
            best_deg = -1
            for n in uncolored:
                sat = len(sat_colors[n])
                deg = degrees[n]
                if sat > best_sat:
                    best_node = n
                    best_sat = sat
                    best_deg = deg
                elif sat == best_sat and deg > best_deg:
                    best_node = n
                    best_deg = deg
                    
            c = 1
            while c in sat_colors[best_node]:
                c += 1
                
            color_assignment[best_node] = c
            uncolored.remove(best_node)
            
            for nbr in adj[best_node]:
                if nbr in uncolored:
                    sat_colors[nbr].add(c)
                    
        num_colors = max(color_assignment.values()) if color_assignment else 0
        deg_sum = sum(degrees.values())
        return num_colors, deg_sum, color_assignment

    # Allow bounds of 10% ratio bounds bounded strictly up to absolute maximal iterations
    allocated_time = max(2.0, min(10.0, args.time_limit * 0.1))
    
    iteration = 0
    # Search initialization heuristics and Local Phase search
    while time.time() - start_time < allocated_time:
        selected = {}
        if iteration == 0:
            for i, cluster in enumerate(clusters):
                selected[i] = min(cluster, key=lambda v: len(adj[v]))
        else:
            for i, cluster in enumerate(clusters):
                selected[i] = random.choice(cluster)
                
        current_selected = dict(selected)
        cur_colors, cur_deg_sum, cur_coloring = evaluate(current_selected)
        
        if cur_colors < global_best_colors:
            global_best_colors = cur_colors
            global_best_selected = dict(current_selected)
            global_best_coloring = cur_coloring
            if logger:
                logger.log(float(global_best_colors))
                
        improved = True
        while improved and time.time() - start_time < allocated_time:
            improved = False
            clusters_order = list(range(num_clusters))
            random.shuffle(clusters_order)
            for i in clusters_order:
                if time.time() - start_time >= allocated_time:
                    break
                cluster = clusters[i]
                if len(cluster) == 1:
                    continue
                original_v = current_selected[i]
                best_v_for_i = original_v
                
                for v in cluster:
                    if v == original_v:
                        continue
                    current_selected[i] = v
                    new_colors, new_deg_sum, new_coloring = evaluate(current_selected)
                    
                    if new_colors < cur_colors or (new_colors == cur_colors and new_deg_sum < cur_deg_sum):
                        cur_colors = new_colors
                        cur_deg_sum = new_deg_sum
                        cur_coloring = new_coloring
                        best_v_for_i = v
                        improved = True
                        
                        if cur_colors < global_best_colors:
                            global_best_colors = cur_colors
                            global_best_selected = dict(current_selected)
                            global_best_coloring = cur_coloring
                            if logger:
                                logger.log(float(global_best_colors))
                                
                if improved:
                    current_selected[i] = best_v_for_i
                else:
                    current_selected[i] = original_v
                    
        iteration += 1

    time_left = args.time_limit - (time.time() - start_time)
    
    best_sol = (global_best_selected, global_best_coloring, global_best_colors)
    
    if time_left > 1.0 and global_best_colors > 1:
        # Provide bounds generated upwards to precision bound solver 
        best_sol = solve_with_gurobi(
            num_vertices, edges_list, num_clusters, clusters, best_sol, logger, time_left
        )
        
    final_sel, final_col, final_obj = best_sol
    
    solution_dict = {
        "objective_value": float(final_obj),
        "selected_vertices": {str(k): v for k, v in final_sel.items()},
        "coloring": {str(k): v for k, v in final_col.items()}
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

if __name__ == "__main__":
    main()