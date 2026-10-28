import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_nodes = instance['num_nodes']
    num_arcs = instance['num_arcs']
    num_commodities = instance['num_commodities']
    num_fleet_types = instance['num_fleet_types']
    nodes = instance['nodes']
    arcs = instance['arcs']
    commodities = instance['commodities']
    fleet_types = instance['fleet_types']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build arc lookup
    arc_list = [(a['from'], a['to']) for a in arcs]
    arc_flow_cost = {(a['from'], a['to']): a['flow_cost'] for a in arcs}
    
    # Build node -> outgoing and incoming arcs
    outgoing = {n: [] for n in nodes}
    incoming = {n: [] for n in nodes}
    for a in arcs:
        outgoing[a['from']].append((a['from'], a['to']))
        incoming[a['to']].append((a['from'], a['to']))
    
    # Create model
    model = gp.Model("MCND")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 1))
    model.setParam('Threads', 1)
    
    # Decision variables
    # Flow variables: x[k, i, j] for commodity k on arc (i,j)
    x = {}
    for k_idx, comm in enumerate(commodities):
        k = comm['id']
        for (i, j) in arc_list:
            x[k, i, j] = model.addVar(
                lb=0.0,
                ub=comm['demand'],  # flow can't exceed demand
                obj=arc_flow_cost[(i, j)] * 1.0,  # unit flow cost (note: flow_cost is per unit of any commodity)
                vtype=GRB.CONTINUOUS,
                name=f"x_{k}_{i}_{j}"
            )
    
    # Vehicle variables: y[v, i, j] for vehicle type v on arc (i,j)
    y = {}
    for ft in fleet_types:
        v = ft['id']
        for (i, j) in arc_list:
            y[v, i, j] = model.addVar(
                lb=0,
                obj=ft['fixed_cost'],
                vtype=GRB.INTEGER,
                name=f"y_{v}_{i}_{j}"
            )
    
    model.update()
    
    # Flow conservation constraints for each commodity at each node
    for comm in commodities:
        k = comm['id']
        origin = comm['origin']
        destination = comm['destination']
        demand = comm['demand']
        
        for n in nodes:
            out_flow = gp.quicksum(x[k, i, j] for (i, j) in outgoing[n] if (k, i, j) in x)
            in_flow = gp.quicksum(x[k, i, j] for (i, j) in incoming[n] if (k, i, j) in x)
            
            if n == origin:
                model.addConstr(out_flow - in_flow == demand, name=f"flow_bal_{k}_{n}")
            elif n == destination:
                model.addConstr(out_flow - in_flow == -demand, name=f"flow_bal_{k}_{n}")
            else:
                model.addConstr(out_flow - in_flow == 0, name=f"flow_bal_{k}_{n}")
    
    # Capacity constraints on each arc
    for (i, j) in arc_list:
        total_flow = gp.quicksum(x[k['id'], i, j] for k in commodities)
        total_cap = gp.quicksum(ft['capacity'] * y[ft['id'], i, j] for ft in fleet_types)
        model.addConstr(total_flow <= total_cap, name=f"cap_{i}_{j}")
    
    # Vehicle balance constraints at each node for each vehicle type
    for ft in fleet_types:
        v = ft['id']
        for n in nodes:
            out_veh = gp.quicksum(y[v, i, j] for (i, j) in outgoing[n])
            in_veh = gp.quicksum(y[v, i, j] for (i, j) in incoming[n])
            model.addConstr(out_veh == in_veh, name=f"veh_bal_{v}_{n}")
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    # Extract solution
    solution = {
        "objective_value": float('inf'),
        "flows": {},
        "vehicles": {}
    }
    
    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        
        for comm in commodities:
            k = comm['id']
            for (i, j) in arc_list:
                val = x[k, i, j].X
                if val > 1e-8:
                    solution["flows"][f"x_{k}_{i}_{j}"] = val
        
        for ft in fleet_types:
            v = ft['id']
            for (i, j) in arc_list:
                val = int(round(y[v, i, j].X))
                if val > 0:
                    solution["vehicles"][f"y_{v}_{i}_{j}"] = val
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)