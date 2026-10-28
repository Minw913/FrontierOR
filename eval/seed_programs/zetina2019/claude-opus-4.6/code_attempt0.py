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
    arcs = instance['arcs']
    fixed_costs = instance['fixed_costs']
    variable_costs = instance['variable_costs']
    commodities = instance['commodities']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build arc list
    arc_keys = []
    arc_from = []
    arc_to = []
    for a in arcs:
        arc_keys.append(f"{a[0]}_{a[1]}")
        arc_from.append(a[0])
        arc_to.append(a[1])
    
    # Build node-arc incidence for efficiency
    # For each node, list of (arc_index, is_outgoing)
    node_out_arcs = [[] for _ in range(num_nodes)]
    node_in_arcs = [[] for _ in range(num_nodes)]
    for idx in range(num_arcs):
        node_out_arcs[arc_from[idx]].append(idx)
        node_in_arcs[arc_to[idx]].append(idx)
    
    # Create model
    model = gp.Model("MCND")
    model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 1))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables
    # y[a] = 1 if arc a is installed
    y = model.addVars(num_arcs, vtype=GRB.BINARY, name="y")
    
    # x[k, a] = fraction of commodity k's demand routed on arc a
    x = {}
    for k in range(num_commodities):
        for a in range(num_arcs):
            vc = variable_costs[arc_keys[a]][k]
            demand = commodities[k]['demand']
            cost = demand * vc
            x[k, a] = model.addVar(lb=0.0, ub=1.0, obj=cost, vtype=GRB.CONTINUOUS,
                                    name=f"x_{k}_{a}")
    
    # Fixed costs for y variables
    for a in range(num_arcs):
        y[a].Obj = fixed_costs[arc_keys[a]]
    
    model.ModelSense = GRB.MINIMIZE
    
    # Flow conservation constraints
    for k in range(num_commodities):
        origin = commodities[k]['origin']
        destination = commodities[k]['destination']
        
        for n in range(num_nodes):
            # outflow - inflow
            expr = gp.LinExpr()
            for a in node_out_arcs[n]:
                expr.add(x[k, a], 1.0)
            for a in node_in_arcs[n]:
                expr.add(x[k, a], -1.0)
            
            if n == origin:
                model.addConstr(expr == 1.0, name=f"flow_{k}_{n}")
            elif n == destination:
                model.addConstr(expr == -1.0, name=f"flow_{k}_{n}")
            else:
                model.addConstr(expr == 0.0, name=f"flow_{k}_{n}")
    
    # Linking constraints: x[k,a] <= y[a]
    for a in range(num_arcs):
        for k in range(num_commodities):
            model.addConstr(x[k, a] <= y[a], name=f"link_{k}_{a}")
    
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
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        open_arcs = {}
        for a in range(num_arcs):
            if y[a].X > 0.5:
                open_arcs[arc_keys[a]] = 1
        
        routings = {}
        for k in range(num_commodities):
            k_routing = {}
            for a in range(num_arcs):
                val = x[k, a].X
                if val > 1e-6:
                    k_routing[arc_keys[a]] = val
            if k_routing:
                routings[str(k)] = k_routing
        
        solution = {
            "objective_value": obj_val,
            "open_arcs": open_arcs,
            "routings": routings
        }
    else:
        solution = {
            "objective_value": float('inf'),
            "open_arcs": {},
            "routings": {}
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)