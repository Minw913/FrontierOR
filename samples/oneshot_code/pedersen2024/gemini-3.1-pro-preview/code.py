import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def prune_tree(selected_edge_ids, selected_nodes_set, node_dict, edge_list):
    selected_edge_ids = set(selected_edge_ids)
    selected_nodes_set = set(selected_nodes_set)
    edge_map = {e['id']: e for e in edge_list}
    
    while True:
        # rebuild adjacency map using current subsets
        adj = {n: {} for n in selected_nodes_set}
        for eid in selected_edge_ids:
            u = edge_map[eid]['from']
            v = edge_map[eid]['to']
            if v not in adj[u]: adj[u][v] = []
            if u not in adj[v]: adj[v][u] = []
            adj[u][v].append(eid)
            adj[v][u].append(eid)
            
        pruned_any = False
        for n in list(selected_nodes_set):
            degree = sum(len(eids) for eids in adj.get(n, {}).values())
            # We prune solely leaf Steiner nodes which add no value beyond routing
            if node_dict[n]['type'] == 'steiner' and degree == 1:
                nbr = list(adj[n].keys())[0]
                brid = adj[n][nbr][0]
                
                selected_nodes_set.remove(n)
                selected_edge_ids.remove(brid)
                pruned_any = True
                
        if not pruned_any:
            break
            
    return list(selected_edge_ids), list(selected_nodes_set)

def main():
    parser = argparse.ArgumentParser(description="Wind Farm Layout Optimization")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to solution output JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions")
    args = parser.parse_args()

    # Read data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    quota = data['parameters']['quota']
    alpha = data['parameters']['alpha']

    nodes = data['nodes']
    edges = data['edges']

    node_dict = {n['id']: n for n in nodes}
    edge_list = edges

    substations = [n['id'] for n in nodes if n['type'] == 'substation']
    pot_turbines = [n['id'] for n in nodes if n['type'] == 'potential_turbine']

    N = len(nodes)
    if substations:
        s0 = substations[0]
    elif nodes:
        s0 = nodes[0]['id']
    else:
        s0 = None

    model = gp.Model("WindFarm")
    
    if args.time_limit > 0:
        model.setParam('TimeLimit', args.time_limit)
        
    model.setParam('Threads', 1)

    if s0 is None:
        # Edge case: entirely empty graph
        out = {"objective_value": 0.0, "selected_turbines": [], "selected_edges": []}
        with open(args.solution_path, 'w') as f:
            json.dump(out, f, indent=2)
        return

    # Variables
    y = {}
    for n_id in node_dict:
        y[n_id] = model.addVar(vtype=GRB.BINARY, name=f"y_{n_id}")

    x = {}
    z = {}
    f_var = {}

    for e in edge_list:
        e_id = e['id']
        u = e['from']
        v = e['to']

        x[e_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{e_id}")
        
        if u != v:
            # Directed routing indicators
            z[(e_id, u, v)] = model.addVar(vtype=GRB.BINARY, name=f"z_{e_id}_{u}_{v}")
            z[(e_id, v, u)] = model.addVar(vtype=GRB.BINARY, name=f"z_{e_id}_{v}_{u}")
            
            # Single-commodity Flow Variables
            f_var[(e_id, u, v)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{e_id}_{u}_{v}")
            f_var[(e_id, v, u)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{e_id}_{v}_{u}")
        else:
            # Trivial constraint for self-loops
            model.addConstr(x[e_id] == 0, name=f"self_loop_{e_id}")

    model.update()

    # 1. Substations are structurally enforced to be part of the final tree
    for s_id in substations:
        model.addConstr(y[s_id] == 1, name=f"substation_{s_id}")

    # 2. Equate edge presence to directed path assignments
    for e in edge_list:
        if e['from'] == e['to']:
            continue
        e_id = e['id']
        u = e['from']
        v = e['to']
        model.addConstr(x[e_id] == z[(e_id, u, v)] + z[(e_id, v, u)], name=f"undir_{e_id}")

    # 3. Restrict boundary for the structural root component
    for e in edge_list:
        if e['from'] == e['to']:
            continue
        e_id = e['id']
        u = e['from']
        v = e['to']
        if v == s0:
            model.addConstr(z[(e_id, u, v)] == 0, name=f"root_in_{e_id}")
        if u == s0:
            model.addConstr(z[(e_id, v, u)] == 0, name=f"root_in_{e_id}")

    # 4. Enforce identical arborescence in-degree configuration for every incorporated node except the root 
    for n_id in node_dict:
        if n_id == s0:
            continue
        arcs_in = []
        for e in edge_list:
            if e['from'] == e['to']: continue
            e_id = e['id']
            if e['to'] == n_id:
                arcs_in.append(z[(e_id, e['from'], n_id)])
            elif e['from'] == n_id:
                arcs_in.append(z[(e_id, e['to'], n_id)])
        model.addConstr(gp.quicksum(arcs_in) == y[n_id], name=f"indegree_{n_id}")

    # 5. Connected Flow Routing
    for n_id in node_dict:
        if n_id == s0:
            continue
        flow_in = []
        flow_out = []
        for e in edge_list:
            if e['from'] == e['to']: continue
            e_id = e['id']
            if e['to'] == n_id:
                flow_in.append(f_var[(e_id, e['from'], n_id)])
                flow_out.append(f_var[(e_id, n_id, e['from'])])
            elif e['from'] == n_id:
                flow_in.append(f_var[(e_id, e['to'], n_id)])
                flow_out.append(f_var[(e_id, n_id, e['to'])])
        model.addConstr(gp.quicksum(flow_in) - gp.quicksum(flow_out) == y[n_id], name=f"flow_bal_{n_id}")

    # 6. Upper bounding limits per flow topology 
    for e in edge_list:
        if e['from'] == e['to']:
            continue
        e_id = e['id']
        u = e['from']
        v = e['to']
        model.addConstr(f_var[(e_id, u, v)] <= (N - 1) * z[(e_id, u, v)])
        model.addConstr(f_var[(e_id, u, v)] >= z[(e_id, u, v)])
        model.addConstr(f_var[(e_id, v, u)] <= (N - 1) * z[(e_id, v, u)])
        model.addConstr(f_var[(e_id, v, u)] >= z[(e_id, v, u)])

    # 7. Constrain endpoints explicitly per inclusion (Edge components restricted exactly on nodes tree subset)
    for e in edge_list:
        if e['from'] == e['to']: continue
        e_id = e['id']
        model.addConstr(x[e_id] <= y[e['from']])
        model.addConstr(x[e_id] <= y[e['to']])

    # 8. Single connected arborescence dimension scale mapping
    model.addConstr(
        gp.quicksum(x[e['id']] for e in edge_list) == gp.quicksum(y[n] for n in node_dict) - 1, 
        name="tree_size"
    )

    # 9. Minimum Power Load Quota Target
    model.addConstr(
        gp.quicksum(node_dict[n]['profit'] * y[n] for n in pot_turbines) >= quota,
        name="energy_quota"
    )

    # 10. Objective Function
    obj_expr = gp.LinExpr()
    
    for n_id in node_dict:
        n_data = node_dict[n_id]
        if n_data['type'] == 'potential_turbine':
            w = alpha * n_data['cost'] + (1 - alpha) * n_data['scenic_impact']
            obj_expr += w * y[n_id]

    for e in edge_list:
        e_id = e['id']
        w = alpha * e['cost'] + (1 - alpha) * e['scenic_impact']
        obj_expr += w * x[e_id]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Attach intermediate solver hook listener
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(cb)

    if model.SolCount > 0:
        raw_selected_nodes = set([n_id for n_id in node_dict if y[n_id].X > 0.5])
        raw_selected_edges_ids = [e['id'] for e in edge_list if x[e['id']].X > 0.5]
        
        # Strip logically redundant steiner-leaves added randomly natively under 0-cost weights 
        pruned_edge_ids, pruned_nodes = prune_tree(raw_selected_edges_ids, raw_selected_nodes, node_dict, edge_list)
        
        final_selected_turbines = [n for n in pruned_nodes if node_dict[n]['type'] == 'potential_turbine']
        final_edges_out = []
        final_obj = 0.0
        
        for n in final_selected_turbines:
            final_obj += alpha * node_dict[n]['cost'] + (1 - alpha) * node_dict[n]['scenic_impact']
        
        edge_map = {e['id']: e for e in edge_list}
        for eid in pruned_edge_ids:
            e = edge_map[eid]
            final_edges_out.append({"from": e['from'], "to": e['to']})
            final_obj += alpha * e['cost'] + (1 - alpha) * e['scenic_impact']
            
        out = {
            "objective_value": final_obj,
            "selected_turbines": final_selected_turbines,
            "selected_edges": final_edges_out
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(out, f, indent=2)
    else:
        print("No feasible solution was observed.", file=sys.stderr)

if __name__ == "__main__":
    main()