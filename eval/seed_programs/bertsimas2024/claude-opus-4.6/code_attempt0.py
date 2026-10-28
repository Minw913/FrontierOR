import argparse
import json
import time
import sys
import numpy as np

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_nodes = data['num_nodes']
    num_commodities = data['num_commodities']
    num_scenarios = data['num_scenarios']
    num_edges = data['num_edges']
    gamma = data['gamma']
    c_0 = data['c_0']
    edges = data['edges']
    construction_costs = data['construction_costs']
    flow_costs = data['flow_costs']
    capacities = data['capacities']
    demands = data['demands']  # [scenario][commodity][node]
    
    import gurobipy as gp
    from gurobipy import GRB
    
    start_time = time.time()
    
    model = gp.Model("network_design")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit - 5)  # Reserve time for output
    
    # Precompute incidence: for each node, which edges are incoming/outgoing
    out_edges = [[] for _ in range(num_nodes)]
    in_edges = [[] for _ in range(num_nodes)]
    for e in range(num_edges):
        u, v = edges[e]
        out_edges[u].append(e)
        in_edges[v].append(e)
    
    # Decision variables
    # z[e]: binary edge activation
    z = model.addVars(num_edges, vtype=GRB.BINARY, name="z")
    
    # x[k,r,e]: flow of commodity k in scenario r on edge e
    x = {}
    for k in range(num_commodities):
        for r in range(num_scenarios):
            for e in range(num_edges):
                x[k, r, e] = model.addVar(lb=0.0, name=f"x[{k},{r},{e}]")
    
    # tf[r,e]: total flow on edge e in scenario r
    tf = {}
    for r in range(num_scenarios):
        for e in range(num_edges):
            tf[r, e] = model.addVar(lb=0.0, name=f"tf[{r},{e}]")
    
    model.update()
    
    # Objective
    # 1) Construction cost
    construction_obj = gp.quicksum(construction_costs[e] * z[e] for e in range(num_edges))
    
    # 2) Expected transportation cost
    inv_S = 1.0 / num_scenarios
    transport_obj = gp.quicksum(
        inv_S * flow_costs[e] * x[k, r, e]
        for k in range(num_commodities)
        for r in range(num_scenarios)
        for e in range(num_edges)
    )
    
    # 3) Expected quadratic congestion penalty: (1/(2*gamma)) * E[sum_e tf[r,e]^2]
    coeff = inv_S / (2.0 * gamma)
    congestion_obj = gp.quicksum(
        coeff * tf[r, e] * tf[r, e]
        for r in range(num_scenarios)
        for e in range(num_edges)
    )
    
    model.setObjective(construction_obj + transport_obj + congestion_obj, GRB.MINIMIZE)
    
    # Constraints
    # 1) Edge activation limit
    model.addConstr(gp.quicksum(z[e] for e in range(num_edges)) <= c_0, "edge_limit")
    
    # 2) Flow conservation: for each commodity k, scenario r, node n
    for k in range(num_commodities):
        for r in range(num_scenarios):
            for n in range(num_nodes):
                outflow = gp.quicksum(x[k, r, e] for e in out_edges[n])
                inflow = gp.quicksum(x[k, r, e] for e in in_edges[n])
                # net outflow - net inflow = supply (positive means supply)
                # demands[r][k][n] is net supply at node n
                model.addConstr(outflow - inflow == demands[r][k][n],
                                f"flow_cons[{k},{r},{n}]")
    
    # 3) Total flow definition: tf[r,e] = sum_k x[k,r,e]
    for r in range(num_scenarios):
        for e in range(num_edges):
            model.addConstr(
                tf[r, e] == gp.quicksum(x[k, r, e] for k in range(num_commodities)),
                f"tf_def[{r},{e}]"
            )
    
    # 4) Capacity and activation: tf[r,e] <= capacities[e] * z[e]
    for r in range(num_scenarios):
        for e in range(num_edges):
            model.addConstr(tf[r, e] <= capacities[e] * z[e],
                            f"cap[{r},{e}]")
    
    # Store best solution found via callback
    best_sol = [None]
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-8:
                best_obj[0] = obj_val
                # Extract solution
                sol = {"objective_value": obj_val, "model_variables": {}, "z": {}}
                
                z_vals = model.cbGetSolution([z[e] for e in range(num_edges)])
                for e in range(num_edges):
                    if z_vals[e] > 0.5:
                        sol["z"][str(e)] = 1
                        sol["model_variables"][f"z[{e}]"] = 1
                
                x_keys = []
                x_vars = []
                for k in range(num_commodities):
                    for r in range(num_scenarios):
                        for e in range(num_edges):
                            x_keys.append((k, r, e))
                            x_vars.append(x[k, r, e])
                
                x_vals = model.cbGetSolution(x_vars)
                for idx, (k, r, e) in enumerate(x_keys):
                    val = x_vals[idx]
                    if abs(val) > 1e-9:
                        sol["model_variables"][f"x[{k},{r},{e}]"] = val
                
                tf_keys = []
                tf_vars = []
                for r in range(num_scenarios):
                    for e in range(num_edges):
                        tf_keys.append((r, e))
                        tf_vars.append(tf[r, e])
                
                tf_vals = model.cbGetSolution(tf_vars)
                for idx, (r, e) in enumerate(tf_keys):
                    val = tf_vals[idx]
                    if abs(val) > 1e-9:
                        sol["model_variables"][f"tf[{r},{e}]"] = val
                
                best_sol[0] = sol
                if logger:
                    logger.log_solution(obj_val, sol)
    
    model.optimize(callback)
    
    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        sol = {"objective_value": obj_val, "model_variables": {}, "z": {}}
        
        for e in range(num_edges):
            val = z[e].X
            if val > 0.5:
                sol["z"][str(e)] = 1
                sol["model_variables"][f"z[{e}]"] = 1
        
        for k in range(num_commodities):
            for r in range(num_scenarios):
                for e in range(num_edges):
                    val = x[k, r, e].X
                    if abs(val) > 1e-9:
                        sol["model_variables"][f"x[{k},{r},{e}]"] = val
        
        for r in range(num_scenarios):
            for e in range(num_edges):
                val = tf[r, e].X
                if abs(val) > 1e-9:
                    sol["model_variables"][f"tf[{r},{e}]"] = val
        
        if logger:
            logger.log_solution(obj_val, sol)
    elif best_sol[0] is not None:
        sol = best_sol[0]
    else:
        sol = {"objective_value": 0.0, "model_variables": {}, "z": {}}
    
    with open(solution_path, 'w') as f:
        json.dump(sol, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)