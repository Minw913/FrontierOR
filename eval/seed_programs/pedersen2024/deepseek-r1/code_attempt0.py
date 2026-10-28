import json
import argparse
from gurobipy import GRB, Model, quicksum
import math
import os
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    parameters = data.get('parameters', {})
    nodes_data = data['nodes']
    edges_data = data['edges']
    
    num_nodes = parameters.get('total_nodes', len(nodes_data))
    num_edges = parameters.get('total_edges', len(edges_data))
    quota = parameters['quota']
    alpha = parameters['alpha']
    
    node_map = {}
    turbine_ids = []
    substation_ids = []
    steiner_ids = []
    all_node_ids = []
    node_energy = {}
    combined_turbine_cost = {}
    for node in nodes_data:
        nid = node['id']
        node_map[nid] = node
        all_node_ids.append(nid)
        if node['type'] == 'potential_turbine':
            turbine_ids.append(nid)
            cost_val = node['cost']
            scenic_val = node['scenic_impact']
            combined_turbine_cost[nid] = alpha * cost_val + (1 - alpha) * scenic_val
            node_energy[nid] = node['profit']
        elif node['type'] == 'substation':
            substation_ids.append(nid)
        elif node['type'] == 'steiner':
            steiner_ids.append(nid)
    
    edge_map = {}
    edge_attrs_by_endpoints = {}
    combined_edge_cost = {}
    for edge in edges_data:
        eid = edge['id']
        u = edge['from']
        v = edge['to']
        key = (min(u, v), max(u, v))
        edge_map[eid] = edge
        edge_attrs_by_endpoints[key] = edge
        cost_val = edge['cost']
        scenic_val = edge['scenic_impact']
        combined_edge_cost[eid] = alpha * cost_val + (1 - alpha) * scenic_val
    
    adj = {nid: [] for nid in all_node_ids}
    for edge in edges_data:
        u = edge['from']
        v = edge['to']
        adj[u].append(v)
        adj[v].append(u)
    
    model = Model("WindFarm")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    
    z = {}
    for nid in all_node_ids:
        z[nid] = model.addVar(vtype=GRB.BINARY, name=f"z_{nid}")
    
    x = {}
    for edge in edges_data:
        eid = edge['id']
        x[eid] = model.addVar(vtype=GRB.BINARY, name=f"x_{eid}")
    
    f = {}
    for edge in edges_data:
        u = edge['from']
        v = edge['to']
        f[(u, v)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{u}_{v}")
        f[(v, u)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{v}_{u}")
    
    for nid in substation_ids:
        model.addConstr(z[nid] == 1, name=f"fix_substation_{nid}")
    
    for edge in edges_data:
        eid = edge['id']
        u = edge['from']
        v = edge['to']
        model.addConstr(x[eid] <= z[u], name=f"edge_node_{eid}_u")
        model.addConstr(x[eid] <= z[v], name=f"edge_node_{eid}_v")
    
    if turbine_ids:
        energy_constr = quicksum(node_energy[i] * z[i] for i in turbine_ids) >= quota
        model.addConstr(energy_constr, name="energy_quota")
    
    if not substation_ids:
        r = None
    else:
        r = substation_ids[0]
    
    total_nodes_var = quicksum(z[i] for i in all_node_ids)
    
    if r is not None:
        outflow_r = quicksum(f[(r, j)] for j in adj[r])
        inflow_r = quicksum(f[(j, r)] for j in adj[r])
        model.addConstr(outflow_r - inflow_r == total_nodes_var - 1, name="root_flow")
    
    for i in all_node_ids:
        if i == r:
            continue
        outflow_i = quicksum(f[(i, j)] for j in adj[i])
        inflow_i = quicksum(f[(j, i)] for j in adj[i])
        model.addConstr(inflow_i - outflow_i == z[i], name=f"flow_cons_{i}")
    
    for edge in edges_data:
        eid = edge['id']
        u = edge['from']
        v = edge['to']
        model.addConstr(f[(u, v)] <= num_nodes * x[eid], name=f"flow_bound_{u}_{v}_{eid}")
        model.addConstr(f[(v, u)] <= num_nodes * x[eid], name=f"flow_bound_{v}_{u}_{eid}")
    
    model.addConstr(quicksum(x[eid] for eid in x) == total_nodes_var - 1, name="tree_size")
    
    obj = quicksum(combined_turbine_cost.get(i, 0) * z[i] for i in all_node_ids)
    obj += quicksum(combined_edge_cost[eid] * x[eid] for eid in x)
    model.setObjective(obj, GRB.MINIMIZE)
    
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()
    
    selected_turbines = []
    selected_edges = []
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            for nid in turbine_ids:
                if z[nid].X > 0.5:
                    selected_turbines.append(nid)
            
            for edge in edges_data:
                eid = edge['id']
                if x[eid].X > 0.5:
                    selected_edges.append({"from": edge['from'], "to": edge['to']})
            
            total_financial = 0.0
            total_scenic = 0.0
            for nid in selected_turbines:
                node = node_map[nid]
                total_financial += node['cost']
                total_scenic += node['scenic_impact']
            
            for edge_dict in selected_edges:
                u = edge_dict['from']
                v = edge_dict['to']
                key = (min(u, v), max(u, v))
                edge_attr = edge_attrs_by_endpoints.get(key)
                if edge_attr:
                    total_financial += edge_attr['cost']
                    total_scenic += edge_attr['scenic_impact']
            
            objective_value = alpha * total_financial + (1 - alpha) * total_scenic
        else:
            objective_value = 0
            selected_turbines = []
            selected_edges = []
    else:
        objective_value = 0
        selected_turbines = []
        selected_edges = []
    
    solution = {
        "objective_value": objective_value,
        "selected_turbines": selected_turbines,
        "selected_edges": selected_edges
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()