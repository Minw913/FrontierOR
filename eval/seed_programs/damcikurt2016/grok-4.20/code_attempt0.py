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

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    gen = data['generator']
    tree = data['scenario_tree']
    nodes = tree['nodes']
    node_list = {n['id']: n for n in nodes}
    root_id = next(n['id'] for n in nodes if n['parent_id'] is None)
    
    # Build ancestor/descendant structures for min up/down
    ancestors = {nid: [] for nid in node_list}
    descendants = {nid: [] for nid in node_list}
    children = {nid: node_list[nid]['children_ids'] for nid in node_list}
    
    def build_ancestors(node_id, anc):
        ancestors[node_id] = anc[:]
        for child_id in children[node_id]:
            build_ancestors(child_id, anc + [node_id])
    
    build_ancestors(root_id, [])
    
    def build_descendants(node_id, desc_list):
        for d in desc_list:
            descendants[d].append(node_id)
        for child_id in children[node_id]:
            build_descendants(child_id, desc_list + [node_id])
    
    build_descendants(root_id, [])
    
    # Parameters
    T = tree['T_num_periods']
    L = tree['L_min_up_time']
    ell = tree['ell_min_down_time']
    Cmin = gen['C_lower_MW']
    Cmax = gen['C_upper_MW']
    Vplus = gen['V_plus_MW_per_h']
    Vminus = gen['V_minus_MW_per_h']
    SuCost = gen['U_bar_startup_cost_dollars']
    SdCost = gen['U_lower_shutdown_cost_dollars']
    a = gen['fuel_a_dollars_per_h']
    b = gen['fuel_b_dollars_per_MWh']
    y0 = tree['initial_generator_status_y0']
    x0 = tree['initial_generation_x0_MW']
    
    # Create model
    m = gp.Model("Generator_Scheduling")
    m.setParam(GRB.Param.Seed, 0)
    m.setParam(GRB.Param.MIPGap, 1e-4)
    m.setParam(GRB.Param.NumericFocus, 0)
    m.setParam(GRB.Param.Threads, 1)
    m.setParam(GRB.Param.TimeLimit, args.time_limit)
    
    # Variables
    node_ids = list(node_list.keys())
    nonroot_ids = [nid for nid in node_ids if nid != root_id]
    
    x = m.addVars(node_ids, lb=0.0, ub=Cmax, name="x", vtype=GRB.CONTINUOUS)
    y = m.addVars(node_ids, lb=0, ub=1, name="y", vtype=GRB.BINARY)
    u = m.addVars(nonroot_ids, lb=0, ub=1, name="u", vtype=GRB.BINARY)
    s = m.addVars(nonroot_ids, lb=0, ub=1, name="s", vtype=GRB.BINARY)  # shutdown indicator
    
    # Objective: maximize expected profit
    obj_expr = gp.LinExpr()
    for n_id in node_ids:
        n = node_list[n_id]
        prob = n['probability']
        price = n['electricity_price_dollars_per_MWh']
        node_x = x[n_id]
        node_y = y[n_id]
        fuel_cost = a * node_y + b * node_x
        revenue = price * node_x
        obj_expr.add(revenue - fuel_cost, prob)
    
    for n_id in nonroot_ids:
        prob = node_list[n_id]['probability']
        obj_expr.add(-SuCost * u[n_id], prob)
        obj_expr.add(-SdCost * s[n_id], prob)
    
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Constraints
    # 1. Generation bounds
    for n_id in node_ids:
        n = node_list[n_id]
        m.addConstr(x[n_id] <= Cmax * y[n_id])
        m.addConstr(x[n_id] >= Cmin * y[n_id])
    
    # 2. Initial conditions
    if y0 == 1:
        m.addConstr(y[root_id] == 1)
        m.addConstr(x[root_id] - x0 <= Vplus)
        m.addConstr(x0 - x[root_id] <= Vminus)
    else:
        m.addConstr(y[root_id] == 0)
        m.addConstr(x[root_id] == 0)
    
    # 3. Startup and shutdown logic + linking
    for n_id in nonroot_ids:
        n = node_list[n_id]
        p_id = n['parent_id']
        m.addConstr(u[n_id] <= y[n_id])
        m.addConstr(u[n_id] <= 1 - y[p_id])
        m.addConstr(u[n_id] >= y[n_id] - y[p_id])
        
        m.addConstr(s[n_id] == y[p_id] - y[n_id] + u[n_id])
    
    # 4. Ramp constraints
    for n_id in nonroot_ids:
        n = node_list[n_id]
        p_id = n['parent_id']
        # When continuing to be on
        m.addConstr(x[n_id] - x[p_id] <= Vplus + Cmax * (1 - y[p_id]))
        m.addConstr(x[p_id] - x[n_id] <= Vminus + Cmax * (1 - y[n_id]))
    
    # 5. Minimum up time
    for n_id in node_ids:
        if L <= 1:
            continue
        n = node_list[n_id]
        t = n['time']
        if t + L - 1 > T:
            continue
        # Find all descendants within L-1 steps
        for d_id in descendants.get(n_id, []):
            d = node_list[d_id]
            if d['time'] - t < L and d['time'] > t:
                m.addConstr(y[d_id] >= u[n_id])
    
    # 6. Minimum down time
    for n_id in node_ids:
        if ell <= 1:
            continue
        n = node_list[n_id]
        t = n['time']
        if t + ell - 1 > T:
            continue
        for d_id in descendants.get(n_id, []):
            d = node_list[d_id]
            if d['time'] - t < ell and d['time'] > t:
                m.addConstr(y[d_id] <= 1 - s[n_id] if n_id != root_id else 1 - (1 - y[n_id]))
    
    # For root shutdown if needed (rare)
    if y0 == 1 and ell > 1:
        for d_id in descendants.get(root_id, []):
            d = node_list[d_id]
            if d['time'] <= ell:
                m.addConstr(y[d_id] <= y[root_id])
    
    # Callback for logging incumbents
    incumbent_obj = -float('inf')
    best_solution = None
    
    def callback(model, where):
        nonlocal incumbent_obj, best_solution
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > incumbent_obj + 1e-6:
                incumbent_obj = obj_val
                sol = {}
                x_sol = {}
                y_sol = {}
                u_sol = {}
                
                for nid in node_ids:
                    yv = model.cbGetSolution(y[nid])
                    xv = model.cbGetSolution(x[nid])
                    y_sol[str(nid)] = int(round(yv))
                    if abs(xv) > 1e-6 or y_sol[str(nid)] == 1:
                        x_sol[str(nid)] = float(xv)
                
                for nid in nonroot_ids:
                    uv = model.cbGetSolution(u[nid])
                    if abs(uv) > 1e-6:
                        u_sol[str(nid)] = int(round(uv))
                
                solution_dict = {
                    "x": x_sol,
                    "y": y_sol,
                    "u": u_sol,
                    "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
                }
                
                if logger:
                    logger.log_solution(obj_val, solution_dict)
                
                best_solution = solution_dict
                best_solution['objective_value'] = float(obj_val)
    
    m.optimize(callback)
    
    # Extract final solution
    if m.status == GRB.INFEASIBLE:
        print("Model is infeasible", file=sys.stderr)
        return
    
    final_obj = m.objVal if m.objVal is not None else 0.0
    solution = {
        "objective_value": float(final_obj),
        "x": {},
        "y": {},
        "u": {},
        "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
    }
    
    for nid in node_ids:
        yval = y[nid].X
        xval = x[nid].X
        y_sol = int(round(yval))
        solution["y"][str(nid)] = y_sol
        if y_sol == 1 or abs(xval) > 1e-5:
            solution["x"][str(nid)] = float(xval)
    
    for nid in nonroot_ids:
        uval = u[nid].X
        if abs(uval) > 1e-5:
            solution["u"][str(nid)] = int(round(uval))
    
    # If we have a better solution from callback, use it
    if best_solution and abs(best_solution.get('objective_value', -float('inf')) - final_obj) < 1e-4:
        solution = best_solution
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()