import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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

    params = instance['parameters']
    nodes = instance['nodes']
    edges = instance['edges']
    quota = params['quota']
    alpha = params['alpha']
    N = params['total_nodes']
    M = params['total_edges']

    # Map node ids to indices 0..N-1
    node_list = sorted(nodes, key=lambda n: n['id'])
    node_id_to_idx = {node['id']: i for i, node in enumerate(node_list)}
    
    turbine_nodes = []
    substation_nodes = []
    steiner_nodes = []
    node_cost = [0.0] * N
    node_profit = [0.0] * N
    node_scenic = [0.0] * N
    node_type = [''] * N
    node_original_id = [0] * N

    for i, node in enumerate(node_list):
        idx = i
        ntype = node['type']
        node_type[idx] = ntype
        node_original_id[idx] = node['id']
        node_cost[idx] = node['cost']
        node_profit[idx] = node['profit']
        node_scenic[idx] = node['scenic_impact']
        if ntype == 'potential_turbine':
            turbine_nodes.append(idx)
        elif ntype == 'substation':
            substation_nodes.append(idx)
        elif ntype == 'steiner':
            steiner_nodes.append(idx)

    # Edges
    edge_list = []
    edge_cost = []
    edge_scenic = []
    edge_from = []
    edge_to = []
    edge_original_id = []  # not really needed
    for e in edges:
        u = node_id_to_idx[e['from']]
        v = node_id_to_idx[e['to']]
        edge_list.append((u, v))
        edge_cost.append(e['cost'])
        edge_scenic.append(e['scenic_impact'])
        edge_from.append(u)
        edge_to.append(v)

    E = len(edge_list)

    # Build model
    model = gp.Model("WindFarmSteinerTree")
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.MIPGap, 0.01)  # reasonable gap
    model.setParam(GRB.Param.LogToConsole, 0)

    # Variables
    y = model.addVars(turbine_nodes, vtype=GRB.BINARY, name="y")  # build turbine
    x = model.addVars(E, vtype=GRB.BINARY, name="x")  # use edge
    u = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0.0, ub=N, name="u")  # MTZ potentials

    # Objective: alpha * cost + (1-alpha) * scenic
    obj_cost = gp.quicksum(edge_cost[i] * x[i] for i in range(E))
    obj_cost += gp.quicksum(node_cost[i] * y[i] for i in turbine_nodes)
    
    obj_scenic = gp.quicksum(edge_scenic[i] * x[i] for i in range(E))
    obj_scenic += gp.quicksum(node_scenic[i] * y[i] for i in turbine_nodes)
    
    objective = alpha * obj_cost + (1.0 - alpha) * obj_scenic
    model.setObjective(objective, GRB.MINIMIZE)

    # Energy quota
    model.addConstr(gp.quicksum(node_profit[i] * y[i] for i in turbine_nodes) >= quota, "quota")

    # All substations must be included
    for s in substation_nodes:
        model.addConstr(u[s] == 0, f"substation_root_{s}")

    # Every node that is used must have degree at least 1 (except if only one substation)
    # But we use MTZ + linking constraints

    # Degree constraints for used nodes
    for i in range(N):
        if node_type[i] == 'potential_turbine':
            used = y[i]
        elif node_type[i] == 'substation':
            used = 1
        else:  # steiner
            used = gp.quicksum(x[j] for j in range(E) if edge_from[j] == i or edge_to[j] == i)
            # We will link via flow or just use MTZ to force connectivity; to tighten:
            model.addConstr(used <= (N-1) * gp.quicksum(x[j] for j in range(E) if edge_from[j]==i or edge_to[j]==i))
            continue  # handled by linking below

        deg = gp.quicksum(x[j] for j in range(E) if edge_from[j] == i or edge_to[j] == i)
        if node_type[i] == 'potential_turbine':
            model.addConstr(deg >= used, f"deg_turb_{i}")
            model.addConstr(deg <= N * used, f"deg_turb_upper_{i}")
        else:
            model.addConstr(deg >= 1, f"deg_sub_{i}")

    # MTZ constraints for connectivity (only on nodes that can be used)
    bigM = N
    for i in range(E):
        u_i = edge_from[i]
        u_j = edge_to[i]
        # Only add MTZ if both sides can be optionally used
        if node_type[u_i] == 'steiner' or node_type[u_i] == 'potential_turbine':
            if node_type[u_j] == 'steiner' or node_type[u_j] == 'potential_turbine':
                model.addConstr(u[u_i] - u[u_j] + bigM * x[i] <= bigM - 1, f"mtz_{i}_1")
                model.addConstr(u[u_j] - u[u_i] + bigM * x[i] <= bigM - 1, f"mtz_{i}_2")
            else:
                # u_j is fixed (substation), already u[u_j]=0
                model.addConstr(u[u_i] + bigM * x[i] <= bigM - 1 + 0, f"mtz_to_sub_{i}")
        elif node_type[u_j] == 'steiner' or node_type[u_j] == 'potential_turbine':
            model.addConstr(u[u_j] + bigM * x[i] <= bigM - 1, f"mtz_from_sub_{i}")

    # Link y to u for turbines: if y=1 then u >=1 (since substations have u=0)
    for t in turbine_nodes:
        model.addConstr(u[t] >= y[t], f"u_link_{t}")

    # Steiner nodes are used only if they have positive degree (implicit via MTZ)

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(float(obj_val))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        objective_value = model.objVal
        selected_turbines = []
        for t in turbine_nodes:
            if y[t].X > 0.5:
                selected_turbines.append(node_original_id[t])
        
        selected_edges = []
        for i in range(E):
            if x[i].X > 0.5:
                u_id = node_original_id[edge_from[i]]
                v_id = node_original_id[edge_to[i]]
                selected_edges.append({"from": int(min(u_id, v_id)), "to": int(max(u_id, v_id))})
        
        # Sort for determinism
        selected_turbines.sort()
        selected_edges = sorted(selected_edges, key=lambda e: (e["from"], e["to"]))
        
        solution = {
            "objective_value": float(objective_value),
            "selected_turbines": selected_turbines,
            "selected_edges": selected_edges
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible or other failure - write empty feasible if possible, but per spec we should provide something
        # For safety output minimal feasible if any, but model should be feasible in most cases
        solution = {
            "objective_value": 0.0,
            "selected_turbines": [],
            "selected_edges": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()