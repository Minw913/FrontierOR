import argparse
import json
import time
import sys
from collections import defaultdict

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    W = instance['bin_capacity']
    items = instance['items']
    num_item_types = instance['num_item_types']
    
    # Sort items by width descending for better performance
    item_list = [(item['type_id'], item['width'], item['demand']) for item in items]
    
    # Build arc-flow graph
    # Nodes: 0 to W
    # Item arcs: (d, d+w_i) for each item type i, for each valid d
    # Loss arcs: (d, d+1) for each d from 0 to W-1 (or we can use loss arcs to W directly)
    
    # To keep the model manageable, we compress the graph by only using "reachable" nodes
    # A node d is reachable if there's a path from 0 to d using item widths and loss arcs
    
    widths = sorted(set(item['width'] for item in items))
    
    # Determine reachable nodes using BFS/dynamic programming
    reachable = set()
    reachable.add(0)
    queue = [0]
    visited = set([0])
    
    while queue:
        new_queue = []
        for d in queue:
            for w in widths:
                nd = d + w
                if nd <= W and nd not in visited:
                    visited.add(nd)
                    new_queue.append(nd)
        queue = new_queue
    
    reachable = visited
    # Also add W (sink) - it should be reachable via loss arcs
    reachable.add(W)
    
    # Now we also need to figure out which nodes can reach W (backward reachability)
    # A node d can reach W if there's a path from d to W
    back_reachable = set()
    back_reachable.add(W)
    bqueue = [W]
    bvisited = set([W])
    
    while bqueue:
        new_bqueue = []
        for d in bqueue:
            for w in widths:
                nd = d - w
                if nd >= 0 and nd not in bvisited and nd in reachable:
                    bvisited.add(nd)
                    new_bqueue.append(nd)
            # Loss arc: (d-1, d) means d-1 can reach d
            # So from d, go back to d-1
            if d - 1 >= 0 and d - 1 not in bvisited:
                bvisited.add(d - 1)
                new_bqueue.append(d - 1)
        bqueue = new_bqueue
    
    back_reachable = bvisited
    
    # Valid nodes are those both forward and backward reachable
    valid_nodes = sorted(reachable & back_reachable)
    valid_set = set(valid_nodes)
    
    # Build arcs
    # Item arcs: for each item type, for each valid start position d where d+w <= W and both d and d+w are valid
    # Loss arcs: for each valid d, (d, next_valid_node) - but standard arc-flow uses (d, d+1) loss arcs
    # Actually, to compress, we use loss arcs that skip gaps: if d is valid and the next valid node is d', 
    # we add a single loss arc (d, d')
    
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("arcflow")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    elapsed = time.time() - start_time
    remaining = max(1, args.time_limit - int(elapsed))
    model.Params.TimeLimit = remaining
    
    # Create variables for item arcs
    # Variable naming: x_{type_id}_{d}_{d+w} for item arcs
    # Variable naming: loss_{d}_{d'} for loss arcs
    
    item_arc_vars = {}  # (type_id, d, d+w) -> var
    loss_arc_vars = {}  # (d, d') -> var
    
    # Group item types by width
    width_to_types = defaultdict(list)
    type_demand = {}
    for tid, w, dem in item_list:
        width_to_types[w].append(tid)
        type_demand[tid] = dem
    
    # For each item type and each valid position d
    for tid, w, dem in item_list:
        for d in valid_nodes:
            if d + w <= W and (d + w) in valid_set:
                var_name = f"x_{tid}_{d}_{d+w}"
                var = model.addVar(vtype=GRB.INTEGER, lb=0, name=var_name)
                item_arc_vars[(tid, d, d + w)] = var
    
    # Loss arcs: connect consecutive valid nodes
    valid_nodes_sorted = sorted(valid_set)
    node_index = {n: i for i, n in enumerate(valid_nodes_sorted)}
    
    for i in range(len(valid_nodes_sorted) - 1):
        d = valid_nodes_sorted[i]
        d_next = valid_nodes_sorted[i + 1]
        var_name = f"loss_{d}_{d_next}"
        var = model.addVar(vtype=GRB.INTEGER, lb=0, name=var_name)
        loss_arc_vars[(d, d_next)] = var
    
    model.update()
    
    # Flow conservation constraints
    # For each node d (except source 0 and sink W):
    #   sum of incoming arcs = sum of outgoing arcs
    # For source: outgoing - incoming = z (number of bins)
    # For sink: incoming - outgoing = z
    
    # We need a variable z for the number of bins (flow value)
    z = model.addVar(vtype=GRB.INTEGER, lb=0, name="z")
    model.update()
    
    # Build incoming and outgoing arc lists for each node
    incoming = defaultdict(list)  # node -> list of (var)
    outgoing = defaultdict(list)  # node -> list of (var)
    
    for (tid, d, dp), var in item_arc_vars.items():
        outgoing[d].append(var)
        incoming[dp].append(var)
    
    for (d, dp), var in loss_arc_vars.items():
        outgoing[d].append(var)
        incoming[dp].append(var)
    
    # Flow conservation
    for node in valid_nodes_sorted:
        in_vars = incoming[node]
        out_vars = outgoing[node]
        
        if node == 0:
            # outgoing - incoming = z
            model.addConstr(
                gp.quicksum(out_vars) - gp.quicksum(in_vars) == z,
                name=f"flow_{node}"
            )
        elif node == W:
            # incoming - outgoing = z
            model.addConstr(
                gp.quicksum(in_vars) - gp.quicksum(out_vars) == z,
                name=f"flow_{node}"
            )
        else:
            # incoming = outgoing
            model.addConstr(
                gp.quicksum(in_vars) == gp.quicksum(out_vars),
                name=f"flow_{node}"
            )
    
    # Demand constraints: for each item type, sum of all its arcs >= demand
    for tid, w, dem in item_list:
        arc_vars_for_type = []
        for d in valid_nodes:
            if (tid, d, d + w) in item_arc_vars:
                arc_vars_for_type.append(item_arc_vars[(tid, d, d + w)])
        if arc_vars_for_type:
            model.addConstr(
                gp.quicksum(arc_vars_for_type) >= dem,
                name=f"demand_{tid}"
            )
    
    model.setObjective(z, GRB.MINIMIZE)
    
    # Compute a quick lower bound: sum(w_i * d_i) / W rounded up
    total_area = sum(w * d for _, w, d in item_list)
    lb = -(-total_area // W)  # ceiling division
    
    # Add callback for logging
    best_obj = [float('inf')]
    best_sol = [None]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                # Extract solution
                sol_dict = {}
                all_vars = model.getVars()
                vals = model.cbGetSolution(all_vars)
                for v, val in zip(all_vars, vals):
                    if abs(val) > 1e-6:
                        sol_dict[v.VarName] = round(val)
                
                best_sol[0] = sol_dict
                
                output = {
                    "objective_value": obj_val,
                    "model_variables": sol_dict
                }
                if logger:
                    logger.log_solution(obj_val, output)
    
    model.optimize(callback)
    
    # Extract final solution
    if model.SolCount > 0:
        obj_val = int(round(model.ObjVal))
        sol_dict = {}
        for v in model.getVars():
            val = v.X
            if abs(val) > 1e-6:
                sol_dict[v.VarName] = round(val)
        
        output = {
            "objective_value": obj_val,
            "model_variables": sol_dict
        }
        if logger:
            logger.log_solution(obj_val, output)
    elif best_sol[0] is not None:
        output = {
            "objective_value": best_obj[0],
            "model_variables": best_sol[0]
        }
    else:
        # Fallback: use one bin per item (worst case feasible solution)
        total_items = sum(d for _, _, d in item_list)
        output = {
            "objective_value": total_items,
            "model_variables": {}
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    solve()