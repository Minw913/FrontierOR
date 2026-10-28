import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except ImportError:
        logger = None

    model = gp.Model("ThermalGenerator")
    
    # Configure Gurobi
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    # Extract data
    gen_data = data['generator']
    tree_data = data['scenario_tree']
    
    C_lower = gen_data['C_lower_MW']
    C_upper = gen_data['C_upper_MW']
    V_plus = gen_data['V_plus_MW_per_h']
    V_minus = gen_data['V_minus_MW_per_h']
    U_start = gen_data['U_bar_startup_cost_dollars']
    U_shut = gen_data['U_lower_shutdown_cost_dollars']
    a = gen_data['fuel_a_dollars_per_h']
    b = gen_data['fuel_b_dollars_per_MWh']
    
    L = tree_data['L_min_up_time']
    ell = tree_data['ell_min_down_time']
    y0 = tree_data['initial_generator_status_y0']
    x0 = tree_data['initial_generation_x0_MW']
    
    nodes = {n['id']: n for n in tree_data['nodes']}
    
    root_id = None
    for nid, n in nodes.items():
        if n['parent_id'] is None:
            root_id = nid
            break
            
    non_root_ids = [nid for nid in nodes.keys() if nid != root_id]
    
    # Decision Variables
    y = model.addVars(nodes.keys(), vtype=GRB.BINARY, name="y")
    x = model.addVars(nodes.keys(), lb=0, name="x")
    u = model.addVars(non_root_ids, lb=0, ub=1, name="u")
    v = model.addVars(non_root_ids, lb=0, ub=1, name="v")
    
    # Constraints
    for nid, n in nodes.items():
        # Generation Limits
        model.addConstr(x[nid] >= C_lower * y[nid], name=f"min_gen_{nid}")
        model.addConstr(x[nid] <= C_upper * y[nid], name=f"max_gen_{nid}")
        
        # Parent context for ramping
        if nid == root_id:
            p_y = y0
            p_x = x0
        else:
            p_id = n['parent_id']
            p_y = y[p_id]
            p_x = x[p_id]
            
        # Ramping Up
        model.addConstr(x[nid] - p_x <= V_plus * p_y + C_upper * (1 - p_y), name=f"ramp_up_{nid}")
        # Ramping Down
        model.addConstr(p_x - x[nid] <= V_minus * y[nid] + C_upper * (1 - y[nid]), name=f"ramp_down_{nid}")
        
        # Startup and Shutdown Definitions (Non-Root Only)
        if nid != root_id:
            model.addConstr(u[nid] - v[nid] == y[nid] - p_y, name=f"status_change_{nid}")
            model.addConstr(u[nid] + v[nid] <= 1, name=f"mutually_exclusive_uv_{nid}")
            
    # Minimum Up and Down Times
    for nid, n in nodes.items():
        t_m = n['time']
        
        # Gather all ancestors including self
        ancestors = []
        curr = nid
        while curr is not None:
            ancestors.append(curr)
            curr = nodes[curr]['parent_id']
            
        # Minimum Up Time Constraint
        startup_sum = []
        for anc in ancestors:
            if anc == root_id:
                continue
            t_a = nodes[anc]['time']
            if t_m - t_a <= L - 1:
                startup_sum.append(u[anc])
        if startup_sum:
            model.addConstr(y[nid] >= gp.quicksum(startup_sum), name=f"min_up_{nid}")
            
        # Minimum Down Time Constraint
        shutdown_sum = []
        for anc in ancestors:
            if anc == root_id:
                continue
            t_a = nodes[anc]['time']
            if t_m - t_a <= ell - 1:
                shutdown_sum.append(v[anc])
        if shutdown_sum:
            model.addConstr(1 - y[nid] >= gp.quicksum(shutdown_sum), name=f"min_down_{nid}")
            
    # Objective Function
    expected_profit = gp.quicksum(
        n['probability'] * (n['electricity_price_dollars_per_MWh'] * x[nid] - a * y[nid] - b * x[nid])
        for nid, n in nodes.items()
    ) - gp.quicksum(
        nodes[nid]['probability'] * (U_start * u[nid] + U_shut * v[nid])
        for nid in non_root_ids
    )
    
    model.setObjective(expected_profit, GRB.MAXIMIZE)
    
    # Callback for Solution Logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_val = model.cbGetSolution(x)
            y_val = model.cbGetSolution(y)
            u_val = model.cbGetSolution(u)
            
            sol_x = {str(nid): x_val[nid] for nid in nodes.keys() if abs(x_val[nid]) > 1e-5}
            sol_y = {str(nid): int(round(y_val[nid])) for nid in nodes.keys() if abs(y_val[nid]) > 0.5}
            sol_u = {str(nid): int(round(u_val[nid])) for nid in non_root_ids if abs(u_val[nid]) > 0.5}
            
            solution = {
                "objective_value": obj,
                "x": sol_x,
                "y": sol_y,
                "u": sol_u,
                "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
            }
            if logger:
                logger.log_solution(obj, solution)
                
    if logger:
        model.optimize(callback)
    else:
        model.optimize()
        
    # Final Solution Output
    if model.SolCount > 0:
        sol_x = {str(nid): x[nid].X for nid in nodes.keys() if abs(x[nid].X) > 1e-5}
        sol_y = {str(nid): int(round(y[nid].X)) for nid in nodes.keys() if abs(y[nid].X) > 0.5}
        sol_u = {str(nid): int(round(u[nid].X)) for nid in non_root_ids if abs(u[nid].X) > 0.5}
        
        final_solution = {
            "objective_value": model.ObjVal,
            "x": sol_x,
            "y": sol_y,
            "u": sol_u,
            "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
        }
    else:
        final_solution = {
            "objective_value": 0.0,
            "x": {},
            "y": {},
            "u": {},
            "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
        }
        
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    solve()