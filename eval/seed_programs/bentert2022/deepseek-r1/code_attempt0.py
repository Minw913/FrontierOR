import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    vertices = instance['vertices']
    
    # Extract coordinates and vertex IDs
    coords = {}
    for v in vertices:
        vid = v['id']
        coords[vid] = (v['x'], v['y'])
    
    # Sort vertex IDs for consistent ordering
    ids = sorted(coords.keys())
    
    # Precompute all edges (complete graph) with integer weights
    edges = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            u = ids[i]
            v = ids[j]
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            weight = (x1 - x2)**2 + (y1 - y2)**2
            weight = int(round(weight))  # Convert to integer
            edges.append((u, v, weight))
    
    # Compute Minimum Spanning Tree (MST) using Kruskal's algorithm
    sorted_edges = sorted(edges, key=lambda x: x[2])
    parent = {id: id for id in ids}
    rank = {id: 0 for id in ids}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True
    
    mst_edges = []
    for edge in sorted_edges:
        u, v, w = edge
        if union(u, v):
            mst_edges.append(edge)
            if len(mst_edges) == n - 1:
                break
    
    # Initialize Gurobi model
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)  # Single core
    
    # Create edge selection variables
    x = {}
    for edge in edges:
        u, v, w = edge
        key = (u, v)
        x[key] = model.addVar(vtype=GRB.BINARY, name=f'x_{u}_{v}')
    
    # Create vertex cost variables
    y = {}
    for vid in ids:
        y[vid] = model.addVar(vtype=GRB.CONTINUOUS, name=f'y_{vid}')
    
    # Create flow variables for connectivity constraints
    flow_vars = {}
    for edge in edges:
        u, v, w = edge
        key = (u, v)
        f_uv = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'f_{u}_{v}_uv')
        f_vu = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'f_{u}_{v}_vu')
        flow_vars[key] = (f_uv, f_vu)
    
    # Flow conservation constraints
    root = ids[0]
    for i in ids:
        flow_in = 0
        flow_out = 0
        for j in ids:
            if j == i:
                continue
            if i < j:
                # Flow from i to j (directed)
                flow_out += flow_vars[(i, j)][0]
                # Flow from j to i (directed)
                if j < i:
                    raise ValueError("Should not happen")
                else:
                    flow_in += flow_vars[(i, j)][1]
            else:
                # i > j: use edge (j, i)
                flow_out += flow_vars[(j, i)][1]
                flow_in += flow_vars[(j, i)][0]
        
        if i == root:
            model.addConstr(flow_out - flow_in == n - 1, f'flow_root_{i}')
        else:
            model.addConstr(flow_in - flow_out == 1, f'flow_node_{i}')
    
    # Flow capacity constraints
    for edge in edges:
        u, v, w = edge
        key = (u, v)
        f_uv, f_vu = flow_vars[key]
        model.addConstr(f_uv + f_vu <= (n - 1) * x[key], f'flow_cap_{u}_{v}')
    
    # Indicator constraints for vertex costs
    for edge in edges:
        u, v, w = edge
        key = (u, v)
        model.addConstr((x[key] == 1) >> (y[u] >= w), f'cost_u_{u}_{v}')
        model.addConstr((x[key] == 1) >> (y[v] >= w), f'cost_v_{u}_{v}')
    
    # Objective: minimize sum of vertex costs
    model.setObjective(gp.quicksum(y[vid] for vid in ids), GRB.MINIMIZE)
    
    # Set MST as initial solution
    for edge in edges:
        u, v, w = edge
        key = (u, v)
        if (u, v, w) in mst_edges or (v, u, w) in mst_edges:
            x[key].Start = 1
        else:
            x[key].Start = 0
    
    max_incident = {vid: 0 for vid in ids}
    for edge in mst_edges:
        u, v, w = edge
        max_incident[u] = max(max_incident[u], w)
        max_incident[v] = max(max_incident[v], w)
    for vid in ids:
        y[vid].Start = max_incident[vid]
    
    # Setup solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Optimization with callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    try:
        if logger:
            model.optimize(callback)
        else:
            model.optimize()
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Extract solution
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            selected_edges = []
            for edge in edges:
                u, v, w = edge
                key = (u, v)
                if x[key].X > 0.5:  # Edge selected
                    selected_edges.append([u, v])
            
            vertex_costs = {}
            total_cost = 0.0
            for vid in ids:
                cost = y[vid].X
                vertex_costs[str(vid)] = cost
                total_cost += cost
            
            solution = {
                "objective_value": total_cost,
                "selected_edges": selected_edges,
                "vertex_costs": vertex_costs
            }
        else:
            # Fallback to MST if no solution found
            solution = create_mst_solution(mst_edges, ids)
    else:
        # Fallback to MST on error
        solution = create_mst_solution(mst_edges, ids)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

def create_mst_solution(mst_edges, ids):
    # Build graph for vertex costs
    incident_weights = {vid: [] for vid in ids}
    for edge in mst_edges:
        u, v, w = edge
        incident_weights[u].append(w)
        incident_weights[v].append(w)
    
    vertex_costs = {}
    total_cost = 0.0
    for vid in ids:
        max_weight = max(incident_weights[vid]) if incident_weights[vid] else 0
        vertex_costs[str(vid)] = max_weight
        total_cost += max_weight
    
    selected_edges = [[u, v] for u, v, w in mst_edges]
    
    return {
        "objective_value": total_cost,
        "selected_edges": selected_edges,
        "vertex_costs": vertex_costs
    }

if __name__ == "__main__":
    main()