import gurobipy as gp
from gurobipy import GRB
import json
import argparse
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_nodes = instance['num_nodes']
    num_commodities = instance['num_commodities']
    num_scenarios = instance['num_scenarios']
    num_edges = instance['num_edges']
    gamma = instance['gamma']
    c_0 = instance['c_0']
    edges = instance['edges']
    construction_costs = instance['construction_costs']
    flow_costs = instance['flow_costs']
    capacities = instance['capacities']
    demands = instance['demands']
    
    outgoing_edges = [[] for _ in range(num_nodes)]
    incoming_edges = [[] for _ in range(num_nodes)]
    for e_idx, (u, v) in enumerate(edges):
        outgoing_edges[u].append(e_idx)
        incoming_edges[v].append(e_idx)
    
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    z = model.addVars(num_edges, vtype=GRB.BINARY, name="z")
    x = model.addVars(num_scenarios, num_commodities, num_edges, lb=0.0, name="x")
    tf = model.addVars(num_scenarios, num_edges, lb=0.0, name="tf")
    
    for r in range(num_scenarios):
        for k in range(num_commodities):
            for i in range(num_nodes):
                outflow = gp.quicksum(x[r, k, e] for e in outgoing_edges[i])
                inflow = gp.quicksum(x[r, k, e] for e in incoming_edges[i])
                model.addConstr(outflow - inflow == demands[r][k][i], 
                               name=f"flow_cons_r{r}_k{k}_n{i}")
    
    for r in range(num_scenarios):
        for e in range(num_edges):
            model.addConstr(tf[r, e] == gp.quicksum(x[r, k, e] for k in range(num_commodities)),
                           name=f"tf_def_r{r}_e{e}")
    
    for r in range(num_scenarios):
        for e in range(num_edges):
            model.addConstr(tf[r, e] <= capacities[e] * z[e], 
                           name=f"cap_r{r}_e{e}")
    
    model.addConstr(gp.quicksum(z[e] for e in range(num_edges)) <= c_0, name="edge_limit")
    
    construction_cost = gp.quicksum(construction_costs[e] * z[e] for e in range(num_edges))
    trans_cost = (1.0 / num_scenarios) * gp.quicksum(flow_costs[e] * tf[r, e] for r in range(num_scenarios) for e in range(num_edges))
    quad_cost = (1.0 / num_scenarios) * (1/(2*gamma)) * gp.quicksum(tf[r, e] * tf[r, e] for r in range(num_scenarios) for e in range(num_edges))
    model.setObjective(construction_cost + trans_cost + quad_cost)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
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
    except gp.GurobiError as e:
        print(f"Optimization error: {e}")
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        model_vars_dict = {}
        for v in model.getVars():
            val = v.X
            if abs(val) > 1e-8:
                model_vars_dict[v.VarName] = val
        
        z_dict = {}
        for e in range(num_edges):
            var_name = f"z[{e}]"
            val = model_vars_dict.get(var_name, 0.0)
            z_dict[str(e)] = int(round(val))
        
        solution_dict = {
            "objective_value": obj_val,
            "model_variables": model_vars_dict,
            "z": z_dict,
            "model_variable_name_patterns": {
                "z[e]": "binary design variable for edge e",
                "x[k,r,e]": "commodity k, scenario r, edge e flow",
                "tf[r,e]": "total scenario flow on edge e"
            }
        }
    else:
        solution_dict = {
            "objective_value": 1e20,
            "model_variables": {},
            "z": {str(e): 0 for e in range(num_edges)},
            "model_variable_name_patterns": {
                "z[e]": "binary design variable for edge e",
                "x[k,r,e]": "commodity k, scenario r, edge e flow",
                "tf[r,e]": "total scenario flow on edge e"
            }
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == "__main__":
    main()