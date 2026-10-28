import sys
import json
import argparse
import os

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

def solve(args):
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    # Setup logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Initialize Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("MCNDP", env=env)
    
    # Configure limits and focuses per problem requirements
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on feasibility due to complexity of ND problems
    
    # Parse sets
    nodes = set(instance.get('nodes', []))
    arcs = instance.get('arcs', [])
    commodities = instance.get('commodities', [])
    fleet_types = instance.get('fleet_types', [])
    
    out_arcs = {}
    in_arcs = {}
    arc_flow_cost = {}
    arc_keys = []
    
    # Process edges robustness: keep the minimal cost in case of multiple edges same endpoints
    for a in arcs:
        u, v = a['from'], a['to']
        nodes.add(u)
        nodes.add(v)
        uv = (u, v)
        if uv not in arc_flow_cost:
            arc_flow_cost[uv] = a['flow_cost']
            arc_keys.append(uv)
        else:
            if a['flow_cost'] < arc_flow_cost[uv]:
                arc_flow_cost[uv] = a['flow_cost']
                
    for n in nodes:
        out_arcs[n] = []
        in_arcs[n] = []
        
    for uv in arc_keys:
        out_arcs[uv[0]].append(uv)
        in_arcs[uv[1]].append(uv)
        
    # Create bulk variables maps for speed
    x_indices = [(k['id'], uv[0], uv[1]) for k in commodities for uv in arc_keys]
    x_obj = {(k['id'], uv[0], uv[1]): arc_flow_cost[uv] for k in commodities for uv in arc_keys}
    x_ub = {(k['id'], uv[0], uv[1]): k['demand'] for k in commodities for uv in arc_keys}
    
    # Continuous flow distributions
    x = model.addVars(x_indices, vtype=GRB.CONTINUOUS, lb=0, ub=x_ub, obj=x_obj, name="x")
    
    y_indices = [(f['id'], uv[0], uv[1]) for f in fleet_types for uv in arc_keys]
    y_obj = {(f['id'], uv[0], uv[1]): f['fixed_cost'] for f in fleet_types for uv in arc_keys}
    
    # Discrete vehicle deployments
    y = model.addVars(y_indices, vtype=GRB.INTEGER, lb=0, obj=y_obj, name="y")
    
    # 1. Commodity Flow Conservation
    for k in commodities:
        kid = k['id']
        orig = k['origin']
        dest = k['destination']
        dem = k['demand']
        for n in nodes:
            out_list = out_arcs[n]
            in_list = in_arcs[n]
            
            rhs = 0
            if n == orig:
                rhs = dem
            elif n == dest:
                rhs = -dem
                
            if out_list or in_list or rhs != 0:
                out_expr = gp.quicksum(x[kid, u, v] for u, v in out_list)
                in_expr = gp.quicksum(x[kid, u, v] for u, v in in_list)
                model.addConstr(out_expr - in_expr == rhs)
                
    # 2. Vehicle Conservation/Balance
    for f in fleet_types:
        fid = f['id']
        for n in nodes:
            out_list = out_arcs[n]
            in_list = in_arcs[n]
            if out_list or in_list:
                out_expr = gp.quicksum(y[fid, u, v] for u, v in out_list)
                in_expr = gp.quicksum(y[fid, u, v] for u, v in in_list)
                model.addConstr(out_expr - in_expr == 0)
                
    # 3. Capacity Bundle Constraints
    for u, v in arc_keys:
        model.addConstr(
            gp.quicksum(x[k['id'], u, v] for k in commodities) <= 
            gp.quicksum(f['capacity'] * y[f['id'], u, v] for f in fleet_types)
        )
        
    def cb(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Execute Model
    model.optimize(cb)
    
    if model.SolCount > 0:
        flows_out = {}
        for k in commodities:
            kid = k['id']
            for u, v in arc_keys:
                val = x[kid, u, v].X
                flows_out[f"x_{kid}_{u}_{v}"] = val if val > 1e-8 else 0.0
                
        vehicles_out = {}
        for f in fleet_types:
            fid = f['id']
            for u, v in arc_keys:
                val = y[fid, u, v].X
                vehicles_out[f"y_{fid}_{u}_{v}"] = int(round(val))
                
        result = {
            "objective_value": model.ObjVal,
            "flows": flows_out,
            "vehicles": vehicles_out
        }
    else:
        # Fallback empty solution format indicating failure to find acceptable bound
        result = {
            "objective_value": float('inf'),
            "flows": {},
            "vehicles": {}
        }
        
    # Flush output 
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    solve(args)