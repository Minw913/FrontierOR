import argparse
import json
import time
import os
from typing import List, Dict, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import math

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

    n = instance['n']
    vertices = instance['vertices']
    # Build distance matrix (squared Euclidean)
    pos = {v['id']: (v['x'], v['y']) for v in vertices}
    node_ids = sorted(pos.keys())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    dist = {}
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            u = node_ids[i]
            v = node_ids[j]
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            d2 = dx*dx + dy*dy
            dist[(u,v)] = d2
            dist[(v,u)] = d2
            edges.append((u,v))

    # Get lower bounds for warm-start and bounding
    lb_dict = {v['id']: v.get('lower_bound', 0.0) for v in vertices}
    
    # Pre-sort possible radii per node
    possible_radii = {}
    for u in node_ids:
        radii = sorted(set([0.0] + [dist[(u,v)] for v in node_ids if v != u]))
        possible_radii[u] = radii

    start_time = time.time()
    best_obj = float('inf')
    best_solution = None
    best_vertex_costs = None

    # Try to find a good feasible solution quickly using a greedy-like approach: sort edges by weight
    sorted_edges = sorted(edges, key=lambda e: dist[e])
    
    # Union-Find for MST like construction but adjusted for bottleneck per node
    def find(parent, x):
        if parent[x] != x:
            parent[x] = find(parent, parent[x])
        return parent[x]
    
    def union(parent, rank, x, y):
        px = find(parent, x)
        py = find(parent, y)
        if px != py:
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True
        return False

    # Kruskal for MST using actual edge weights - this gives a spanning tree
    parent = {nid: nid for nid in node_ids}
    rank = {nid: 0 for nid in node_ids}
    mst_edges = []
    max_edge = 0.0
    for e in sorted_edges:
        u, v = e
        if union(parent, rank, u, v):
            mst_edges.append(e)
            w = dist[e]
            if w > max_edge:
                max_edge = w
            if len(mst_edges) == n-1:
                break

    # Compute vertex costs for this tree
    inc_tree = {u: [] for u in node_ids}
    for u,v in mst_edges:
        inc_tree[u].append(dist[(u,v)])
        inc_tree[v].append(dist[(u,v)])
    
    vertex_cost = {}
    total_cost = 0.0
    for u in node_ids:
        if inc_tree[u]:
            vertex_cost[u] = max(inc_tree[u])
        else:
            vertex_cost[u] = 0.0
        total_cost += vertex_cost[u]
    
    if total_cost < best_obj:
        best_obj = total_cost
        best_solution = mst_edges[:]
        best_vertex_costs = {str(k): float(vertex_cost[k]) for k in vertex_cost}
        if logger:
            logger.log(float(best_obj))
    
    # Try to improve with a simple local search: try to reduce max radius for high cost nodes
    # This is a heuristic to get better upper bound quickly
    current_edges = mst_edges[:]
    current_cost = best_obj
    improved = True
    while improved and (time.time() - start_time < args.time_limit * 0.4):
        improved = False
        # For each node, see if we can replace its max edge with smaller ones
        inc = {u: [] for u in node_ids}
        for u,v in current_edges:
            w = dist[(u,v)]
            inc[u].append((w, v))
            inc[v].append((w, u))
        
        for u in node_ids:
            if not inc[u]:
                continue
            inc[u].sort(reverse=True)
            max_w = inc[u][0][0]
            if max_w <= lb_dict[u]:
                continue
            # Try to find alternative connections with smaller max
            target_max = max_w * 0.8  # aggressive reduction
            # Find all nodes reachable with edges <= target_max
            parent2 = {nid: nid for nid in node_ids}
            rank2 = {nid: 0 for nid in node_ids}
            for e in edges:
                if dist[e] <= target_max:
                    union(parent2, rank2, e[0], e[1])
            # If still connected to all, we could potentially use only edges <= target_max
            root = find(parent2, u)
            can_reduce = all(find(parent2, v) == root for v in node_ids)
            if can_reduce:
                # Rebuild a spanning tree using only edges <= target_max
                parent3 = {nid: nid for nid in node_ids}
                rank3 = {nid: 0 for nid in node_ids}
                new_edges = []
                sorted_small = [e for e in sorted_edges if dist[e] <= target_max]
                for e in sorted_small:
                    if union(parent3, rank3, e[0], e[1]):
                        new_edges.append(e)
                        if len(new_edges) == n-1:
                            break
                if len(new_edges) == n-1:
                    # Compute new cost
                    inc_new = {u: [] for u in node_ids}
                    for a,b in new_edges:
                        w = dist[(a,b)]
                        inc_new[a].append(w)
                        inc_new[b].append(w)
                    new_vertex_cost = {u: max(inc_new[u]) if inc_new[u] else 0.0 for u in node_ids}
                    new_total = sum(new_vertex_cost.values())
                    if new_total < current_cost - 1e-6:
                        current_edges = new_edges[:]
                        current_cost = new_total
                        improved = True
                        if current_cost < best_obj:
                            best_obj = current_cost
                            best_solution = current_edges[:]
                            best_vertex_costs = {str(k): float(new_vertex_cost[k]) for k in new_vertex_cost}
                            if logger:
                                logger.log(float(best_obj))
                        break  # restart after improvement
    
    # Now use MIP with Gurobi with time limit
    remaining_time = max(1, args.time_limit - int(time.time() - start_time) - 5)
    try:
        model = gp.Model("MinimumPowerSpanningTree")
        model.setParam(GRB.Param.TimeLimit, remaining_time)
        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.MIPGap, 0.01)  # Allow 1% gap if needed
        model.setParam(GRB.Param.LogToConsole, 0)
        
        # Variables: for each possible edge, binary whether selected
        edge_vars = {}
        for u, v in edges:
            edge_vars[(u,v)] = model.addVar(vtype=GRB.BINARY, name=f"e_{u}_{v}")
        
        # For each node, the cost it pays (max edge weight used)
        cost_vars = {}
        for u in node_ids:
            cost_vars[u] = model.addVar(lb=lb_dict[u], ub=max(dist[(u,v)] for v in node_ids if v != u),
                                       vtype=GRB.CONTINUOUS, name=f"c_{u}")
        
        # Objective
        model.setObjective(gp.quicksum(cost_vars[u] for u in node_ids), GRB.MINIMIZE)
        
        # Constraints
        # Each cost_var must be at least the weight of each selected incident edge
        M = 1e9  # big-M, max possible squared distance is usually < 1e9 in practice
        for u in node_ids:
            for v in node_ids:
                if u == v: continue
                ekey = (min(u,v), max(u,v))
                w = dist[(u,v)]
                model.addConstr(cost_vars[u] >= w * edge_vars[ekey] - M * (1 - edge_vars[ekey]))
        
        # Connectivity: use flow or cut-set. Use multi-commodity flow for denser graphs or MTZ like
        # Since n can be up to ~100, use standard subtour elimination with lazy constraints or flow
        # For efficiency we use single commodity flow with node capacities implicitly via edge selection
        # Actually for spanning tree like, we can use flow from root with n-1 units
        root = node_ids[0]
        flow_vars = {}
        for u, v in edges:
            flow_vars[(u,v)] = model.addVar(lb=0, ub=n, vtype=GRB.CONTINUOUS, name=f"f_{u}_{v}")
            flow_vars[(v,u)] = model.addVar(lb=0, ub=n, vtype=GRB.CONTINUOUS, name=f"f_{v}_{u}")
        
        # Flow conservation
        for u in node_ids:
            if u == root: continue
            model.addConstr(
                gp.quicksum(flow_vars[(v,u)] for v in node_ids if v != u) - 
                gp.quicksum(flow_vars[(u,v)] for v in node_ids if v != u) == 1
            )
        
        # Link flow to edge selection
        for u, v in edges:
            e_var = edge_vars[(u,v)]
            model.addConstr(flow_vars[(u,v)] <= (n-1) * e_var)
            model.addConstr(flow_vars[(v,u)] <= (n-1) * e_var)
        
        # At root we send out n-1
        model.addConstr(
            gp.quicksum(flow_vars[(root, v)] for v in node_ids if v != root) - 
            gp.quicksum(flow_vars[(v, root)] for v in node_ids if v != root) == n-1
        )
        
        # Optional: symmetry breaking - force lower indexed nodes to connect earlier but may not help
        # Warm start with best solution found so far
        if best_solution is not None:
            for u,v in edges:
                edge_vars[(u,v)].Start = 0.0
            for u,v in best_solution:
                edge_vars[(min(u,v), max(u,v))].Start = 1.0
            for u in node_ids:
                cost_vars[u].Start = best_vertex_costs.get(str(u), 0.0)
        
        # Callback to log new incumbents
        def callback(model, where):
            nonlocal best_obj, best_solution, best_vertex_costs
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if obj < best_obj - 1e-4:
                    best_obj = obj
                    # Extract solution
                    sol_edges = []
                    sol_costs = {}
                    for u,v in edges:
                        if model.cbGetSolution(edge_vars[(u,v)]) > 0.5:
                            sol_edges.append([int(u), int(v)])
                    for u in node_ids:
                        sol_costs[str(u)] = float(model.cbGetSolution(cost_vars[u]))
                    best_solution = sol_edges
                    best_vertex_costs = sol_costs
                    if logger:
                        logger.log(float(obj))
        
        model.optimize(callback)
        
        # Update with MIP solution if better
        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            if model.SolCount > 0:
                obj_val = model.ObjVal
                if obj_val < best_obj - 1e-4:
                    best_obj = obj_val
                    sol_edges = []
                    sol_cost_dict = {}
                    for u,v in edges:
                        if edge_vars[(u,v)].X > 0.5:
                            sol_edges.append([int(u), int(v)])
                    for u in node_ids:
                        sol_cost_dict[str(u)] = float(cost_vars[u].X)
                    best_solution = sol_edges
                    best_vertex_costs = sol_cost_dict
                    if logger:
                        logger.log(float(best_obj))
        
    except Exception as e:
        # If MIP fails for any reason, fall back to heuristic solution
        pass
    
    # Final output
    if best_solution is None:
        # Last resort: return a star topology to root 0
        best_solution = []
        best_vertex_costs = {}
        root = node_ids[0]
        max_w = 0.0
        for i in range(1, n):
            u = node_ids[i]
            w = dist[(root, u)]
            best_solution.append([int(root), int(u)])
            if w > max_w:
                max_w = w
        for u in node_ids:
            if u == root:
                best_vertex_costs[str(u)] = float(max_w)
            else:
                best_vertex_costs[str(u)] = float(dist[(root, u)])
        best_obj = sum(best_vertex_costs.values())
    
    result = {
        "objective_value": float(best_obj),
        "selected_edges": [[int(u), int(v)] for u,v in best_solution],
        "vertex_costs": best_vertex_costs
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()