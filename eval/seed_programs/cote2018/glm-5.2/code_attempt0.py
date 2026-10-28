import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def set_gurobi_params(m, time_limit=None):
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

def solve_arc_flow_direct(instance, time_limit, logger):
    W = instance['bin_capacity']
    items = instance['items']
    num_items = len(items)
    
    m = gp.Model()
    set_gurobi_params(m, time_limit)
    
    x = {}
    for u in range(W + 1):
        for idx, item in enumerate(items):
            if u + item['width'] <= W:
                x[(u, idx)] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{u}_{item['type_id']}")
    
    l = {}
    for u in range(W):
        l[u] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"l_{u}")
        
    r = m.addVar(vtype=GRB.INTEGER, lb=0, name="r")
    
    # Flow conservation
    # Node 0
    m.addConstr(r == gp.quicksum(x[(0, idx)] for idx in range(num_items) if (0, idx) in x) + l[0])
    
    # Nodes 1 to W-1
    for u in range(1, W):
        in_flow = gp.quicksum(x[(u - item['width'], idx)] for idx, item in enumerate(items) if (u - item['width'], idx) in x) + l[u-1]
        out_flow = gp.quicksum(x[(u, idx)] for idx in range(num_items) if (u, idx) in x) + l[u]
        m.addConstr(in_flow == out_flow)
        
    # Node W
    in_flow = gp.quicksum(x[(W - item['width'], idx)] for idx, item in enumerate(items) if (W - item['width'], idx) in x) + l[W-1]
    m.addConstr(in_flow == r)
    
    # Demand
    for idx, item in enumerate(items):
        m.addConstr(gp.quicksum(x[(u, idx)] for u in range(W - item['width'] + 1)) >= item['demand'])
        
    m.setObjective(r, GRB.MINIMIZE)
    m.optimize()
    
    if m.SolCount == 0:
        return None, None
        
    obj_val = int(round(m.objVal))
    model_vars = {}
    for (u, idx), var in x.items():
        if var.X > 1e-6:
            model_vars[var.VarName] = int(round(var.X))
    for u, var in l.items():
        if var.X > 1e-6:
            model_vars[var.VarName] = int(round(var.X))
    if r.X > 1e-6:
        model_vars[r.VarName] = int(round(r.X))
        
    return obj_val, model_vars

def solve_csp_colgen(instance, time_limit, logger):
    W = instance['bin_capacity']
    items = instance['items']
    num_items = len(items)
    widths = [item['width'] for item in items]
    demands = [item['demand'] for item in items]
    type_ids = [item['type_id'] for item in items]

    # Initial patterns
    patterns = []
    for i in range(num_items):
        p = [0] * num_items
        p[i] = W // widths[i]
        if p[i] == 0:
            p[i] = 1
        patterns.append(p)

    start_time = time.time()
    
    # Column generation loop
    while True:
        if time.time() - start_time > time_limit * 0.5:
            break
            
        m = gp.Model()
        set_gurobi_params(m)
        x = m.addVars(len(patterns), lb=0, name="x")
        demand_constrs = {}
        for i in range(num_items):
            demand_constrs[i] = m.addConstr(gp.quicksum(patterns[p][i] * x[p] for p in range(len(patterns))) >= demands[i])
        m.setObjective(gp.quicksum(x[p] for p in range(len(patterns))), GRB.MINIMIZE)
        m.optimize()
        
        duals = [demand_constrs[i].Pi for i in range(num_items)]
        
        # Subproblem (Knapsack)
        sub = gp.Model()
        set_gurobi_params(sub)
        y = sub.addVars(num_items, vtype=GRB.INTEGER, lb=0, name="y")
        sub.addConstr(gp.quicksum(widths[i] * y[i] for i in range(num_items)) <= W)
        sub.setObjective(gp.quicksum(duals[i] * y[i] for i in range(num_items)), GRB.MAXIMIZE)
        sub.optimize()
        
        if sub.objVal > 1 + 1e-5:
            patterns.append([int(y[i].X + 0.5) for i in range(num_items)])
        else:
            break

    # Final restricted MIP
    remaining_time = max(1, time_limit - (time.time() - start_time))
    m = gp.Model()
    set_gurobi_params(m, remaining_time)
    x = m.addVars(len(patterns), vtype=GRB.INTEGER, lb=0, name="x")
    for i in range(num_items):
        m.addConstr(gp.quicksum(patterns[p][i] * x[p] for p in range(len(patterns))) >= demands[i])
    m.setObjective(gp.quicksum(x[p] for p in range(len(patterns))), GRB.MINIMIZE)
    m.optimize()
    
    if m.SolCount == 0:
        return None, None

    pattern_counts = {}
    for p in range(len(patterns)):
        val = int(round(x[p].X))
        if val > 0:
            pattern_counts[p] = val

    obj_val = int(round(m.objVal))
    
    # Build arc-flow witness
    model_vars = {}
    r_val = 0
    for p, count in pattern_counts.items():
        pattern = patterns[p]
        r_val += count
        curr_pos = 0
        for i in range(num_items):
            for _ in range(pattern[i]):
                var_name = f"x_{curr_pos}_{type_ids[i]}"
                model_vars[var_name] = model_vars.get(var_name, 0) + count
                curr_pos += widths[i]
        for u in range(curr_pos, W):
            var_name = f"l_{u}"
            model_vars[var_name] = model_vars.get(var_name, 0) + count
    model_vars["r"] = r_val
    
    return obj_val, model_vars

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

    W = instance['bin_capacity']
    num_items = instance['num_item_types']
    
    # Decide method
    # If the graph is small enough, solve direct arc-flow
    if num_items * W <= 200000 and W <= 10000:
        obj_val, model_vars = solve_arc_flow_direct(instance, args.time_limit, logger)
    else:
        obj_val, model_vars = solve_csp_colgen(instance, args.time_limit, logger)

    if logger and obj_val is not None:
        solution_dict = {
            "objective_value": obj_val,
            "model_variables": model_vars
        }
        logger.log_solution(obj_val, solution_dict)

    solution = {
        "objective_value": obj_val,
        "model_variables": model_vars
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()