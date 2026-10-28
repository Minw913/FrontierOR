import sys
import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="One-dimensional Cutting Stock / Bin Packing using Restricted Arc-Flow")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to logging JSONL file")
    args = parser.parse_args()

    start_time = time.time()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    W = data["bin_capacity"]
    items = data["items"]
    
    # Initialize Logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    M = len(items)
    if M == 0 or sum(item["demand"] for item in items) == 0:
        solution = {"objective_value": 0, "model_variables": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(0, {})
        return

    widths = [item["width"] for item in items]
    demands = [item["demand"] for item in items]
    type_ids = [item["type_id"] for item in items]

    # Precompute full reachable subset sums to determine graph scale
    reachable = [False] * (W + 1)
    reachable[0] = True
    unique_widths = list(set(widths))
    for i in range(W + 1):
        if reachable[i]:
            for w in unique_widths:
                if i + w <= W:
                    reachable[i + w] = True

    V_full = [i for i, r in enumerate(reachable) if r]
    
    num_arcs = 0
    for u in V_full:
        for w in widths:
            if u + w <= W and reachable[u + w]:
                num_arcs += 1

    # Decide whether to use Full Arc-Flow or dynamically Restricted Arc-Flow via Column Generation
    use_full_arc_flow = (num_arcs <= 200000)

    # 1. Base Heuristic (First-Fit Decreasing)
    flat_items = []
    for i, t_id in enumerate(type_ids):
        flat_items.extend([i] * demands[i])
    flat_items.sort(key=lambda x: widths[x], reverse=True)
    
    bins = []
    for item in flat_items:
        placed = False
        for b in bins:
            if sum(widths[idx] for idx in b) + widths[item] <= W:
                b.append(item)
                placed = True
                break
        if not placed:
            bins.append([item])

    ffd_patterns = []
    for b in bins:
        pat = {}
        for idx in b:
            pat[idx] = pat.get(idx, 0) + 1
        ffd_patterns.append(pat)

    V = set()
    if use_full_arc_flow:
        V = set(V_full)
    
    V.add(0)
    V.add(W)

    # Ensure all FFD nodes are reachable in the graph V
    for pat in ffd_patterns:
        u = 0
        sorted_items = sorted(pat.keys(), key=lambda i: type_ids[i], reverse=True)
        for i in sorted_items:
            for _ in range(pat[i]):
                v_nxt = u + widths[i]
                V.add(u)
                V.add(v_nxt)
                u = v_nxt

    af_start = {}
    best_obj = len(bins)
    best_patterns = ffd_patterns

    # 2. Pattern Generation Phase (CG)
    if not use_full_arc_flow:
        mp = gp.Model("MP")
        mp.setParam('OutputFlag', 0)
        
        patterns = []
        for i in range(M):
            pat = {i: max(1, W // widths[i])}
            patterns.append(pat)
            
        x_vars = []
        for j, pat in enumerate(patterns):
            x_vars.append(mp.addVar(obj=1.0, name=f"p_{j}"))
            
        constrs = []
        for i in range(M):
            expr = gp.quicksum(patterns[j].get(i, 0) * x_vars[j] for j in range(len(patterns)))
            constrs.append(mp.addConstr(expr >= demands[i], name=f"d_{i}"))
            
        kp = gp.Model("KP")
        kp.setParam('OutputFlag', 0)
        y = kp.addVars(M, vtype=GRB.INTEGER, name="y")
        kp.addConstr(gp.quicksum(widths[i] * y[i] for i in range(M)) <= W)
        
        cg_start = time.time()
        while time.time() - cg_start < (args.time_limit * 0.4):
            mp.optimize()
            if mp.status != GRB.OPTIMAL:
                break
                
            pi = [c.Pi for c in constrs]
            kp.setObjective(gp.quicksum(pi[i] * y[i] for i in range(M)), GRB.MAXIMIZE)
            kp.optimize()
            
            if kp.status != GRB.OPTIMAL or kp.ObjVal <= 1.0 + 1e-6:
                break
                
            new_pat = {}
            for i in range(M):
                val = int(round(y[i].X))
                if val > 0:
                    new_pat[i] = val
                    
            if new_pat in patterns:
                break
                
            patterns.append(new_pat)
            col = gp.Column()
            for i in range(M):
                if new_pat.get(i, 0) > 0:
                    col.addTerms(new_pat[i], constrs[i])
            x_vars.append(mp.addVar(obj=1.0, column=col, name=f"p_{len(patterns)-1}"))
            
        # Collect nodes from generated patterns
        for pat in patterns:
            u = 0
            sorted_items = sorted(pat.keys(), key=lambda i: type_ids[i], reverse=True)
            for i in sorted_items:
                for _ in range(pat[i]):
                    v_nxt = u + widths[i]
                    V.add(u)
                    V.add(v_nxt)
                    u = v_nxt

        # Solve Master Problem exactly as an Integer Program to get high quality bound
        for v in mp.getVars():
            v.vtype = GRB.INTEGER
        mp.optimize()
        if mp.SolCount > 0 and mp.ObjVal < best_obj:
            best_obj = int(round(mp.ObjVal))
            best_patterns = []
            for j, pat in enumerate(patterns):
                val = int(round(x_vars[j].X))
                for _ in range(val):
                    best_patterns.append(pat)

    # 3. Create Valid Initial Flow Mapping
    V = sorted(list(V))
    V_set = set(V)

    af_start["z"] = len(best_patterns)
    for pat in best_patterns:
        u = 0
        sorted_items = sorted(pat.keys(), key=lambda i: type_ids[i], reverse=True)
        for i in sorted_items:
            for _ in range(pat[i]):
                v_nxt = u + widths[i]
                af_start[(u, v_nxt, type_ids[i])] = af_start.get((u, v_nxt, type_ids[i]), 0) + 1
                u = v_nxt
        curr = u
        idx = V.index(curr)
        while V[idx] < W:
            nxt_node = V[idx+1]
            af_start[("loss", V[idx], nxt_node)] = af_start.get(("loss", V[idx], nxt_node), 0) + 1
            idx += 1

    # 4. Construct and Solve Restricted Arc-Flow Model
    af = gp.Model("ArcFlow")
    af.setParam('OutputFlag', 0)
    af.setParam('Seed', 0)
    af.setParam('MIPGap', 1e-4)
    af.setParam('NumericFocus', 0)
    af.setParam('Threads', 1)
    
    time_left = args.time_limit - (time.time() - start_time)
    af.setParam('TimeLimit', max(5.0, time_left))

    x = {}
    for u in V:
        for i, w in enumerate(widths):
            if u + w in V_set:
                var = af.addVar(vtype=GRB.INTEGER, name=f"x[{u},{u+w},{type_ids[i]}]")
                if (u, u+w, type_ids[i]) in af_start:
                    var.Start = af_start[(u, u+w, type_ids[i])]
                x[u, u+w, type_ids[i]] = var
                
    loss = {}
    for i in range(len(V)-1):
        u = V[i]
        v_nxt = V[i+1]
        var = af.addVar(vtype=GRB.INTEGER, name=f"loss[{u},{v_nxt}]")
        if ("loss", u, v_nxt) in af_start:
            var.Start = af_start[("loss", u, v_nxt)]
        loss[u, v_nxt] = var
        
    z = af.addVar(vtype=GRB.INTEGER, obj=1.0, name="z")
    if "z" in af_start:
        z.Start = af_start["z"]
        
    # Flow conservation constraints
    for i, u in enumerate(V):
        inflow_terms = []
        for j, w in enumerate(widths):
            u_prev = u - w
            if u_prev in V_set and (u_prev, u, type_ids[j]) in x:
                inflow_terms.append(x[u_prev, u, type_ids[j]])
        inflow = gp.quicksum(inflow_terms)
        if i > 0:
            inflow += loss[V[i-1], u]
        if u == 0:
            inflow += z
            
        outflow_terms = []
        for j, w in enumerate(widths):
            u_next = u + w
            if u_next in V_set and (u, u_next, type_ids[j]) in x:
                outflow_terms.append(x[u, u_next, type_ids[j]])
        outflow = gp.quicksum(outflow_terms)
        if i < len(V) - 1:
            outflow += loss[u, V[i+1]]
        if u == W:
            outflow += z
            
        af.addConstr(inflow == outflow, name=f"flow_{u}")
        
    # Demand satisfaction
    for i, t_id in enumerate(type_ids):
        af.addConstr(gp.quicksum(x[u, u+widths[i], t_id] for u in V if (u, u+widths[i], t_id) in x) >= demands[i], name=f"demand_{t_id}")

    af._vars = af.getVars()
    for var in af._vars:
        var._name = var.VarName

    # 5. Optimization with Callback for Logging
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    x_vals = model.cbGetSolution(model._vars)
                    sol_dict = {}
                    for var, val in zip(model._vars, x_vals):
                        ival = int(round(val))
                        if ival > 0:
                            sol_dict[var._name] = ival
                    logger.log_solution(int(round(obj)), sol_dict)
                except Exception:
                    logger.log(int(round(obj)))

    af.optimize(cb)

    # 6. Extract the Complete Final Witness
    model_variables = {}
    obj_val = 0
    if af.SolCount > 0:
        obj_val = int(round(af.ObjVal))
        for var in af._vars:
            val = int(round(var.X))
            if val > 0:
                model_variables[var._name] = val
    else:
        # Fallback entirely driven by robust initialization flows if Gurobi times out prematurely
        obj_val = int(best_obj)
        for k, v in af_start.items():
            if k == "z":
                model_variables["z"] = v
            elif k[0] == "loss":
                model_variables[f"loss[{k[1]},{k[2]}]"] = v
            else:
                model_variables[f"x[{k[0]},{k[1]},{k[2]}]"] = v

    if logger:
        logger.log_solution(obj_val, model_variables)
        
    solution = {
        "objective_value": obj_val,
        "model_variables": model_variables
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()