import json
import argparse
import gc
import traceback
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    try:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
        except ImportError:
            logger = None

        with open(args.instance_path, 'r') as f:
            data = json.load(f)
            
        gen_data = data['generator']
        tree_data = data['scenario_tree']
        
        C_lower = float(gen_data['C_lower_MW'])
        C_upper = float(gen_data['C_upper_MW'])
        # Cap ramping rates at C_upper to prevent numerical/Big-M issues
        V_plus = min(float(gen_data['V_plus_MW_per_h']), C_upper)
        V_minus = min(float(gen_data['V_minus_MW_per_h']), C_upper)
        U_start = float(gen_data['U_bar_startup_cost_dollars'])
        U_shut = float(gen_data['U_lower_shutdown_cost_dollars'])
        a = float(gen_data['fuel_a_dollars_per_h'])
        b = float(gen_data['fuel_b_dollars_per_MWh'])
        
        L = int(tree_data['L_min_up_time'])
        ell = int(tree_data['ell_min_down_time'])
        y0 = int(tree_data['initial_generator_status_y0'])
        x0 = float(tree_data['initial_generation_x0_MW'])
        
        nodes = tree_data['nodes']
        N = len(nodes)
        
        id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
        
        parent_idx = [None] * N
        prob = np.zeros(N)
        price = np.zeros(N)
        node_ids = np.zeros(N, dtype=np.int64)
        
        root_idx = None
        for i, n in enumerate(nodes):
            node_ids[i] = n['id']
            prob[i] = float(n['probability'])
            price[i] = float(n['electricity_price_dollars_per_MWh'])
            pid = n['parent_id']
            if pid is None:
                root_idx = i
            else:
                parent_idx[i] = id_to_idx[pid]
                
        # Clean up JSON dictionaries to aggressively free memory
        del data
        del tree_data
        del gen_data
        del nodes
        gc.collect()

        # Build sparse matrices for efficient constraint generation
        row_P = []
        col_P = []
        val_P = []
        for i in range(N):
            if parent_idx[i] is not None:
                row_P.append(i)
                col_P.append(parent_idx[i])
                val_P.append(1.0)
                
        P = sp.csr_matrix((val_P, (row_P, col_P)), shape=(N, N))
        I = sp.eye(N, format='csr')
        
        M_nonroot = sp.diags([1.0 if i != root_idx else 0.0 for i in range(N)], format='csr')
        M_diff = M_nonroot - P

        x0_vec = np.zeros(N)
        x0_vec[root_idx] = x0

        y0_vec = np.zeros(N)
        y0_vec[root_idx] = y0

        ones_vec = np.ones(N)

        # Initialize Gurobi natively to avoid dbus/license check errors on empty init
        model = gp.Model("ThermalGenerator")
        model.setParam('OutputFlag', 1)
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', args.time_limit)
        
        # Add Variables using Matrix Variables for immense memory savings
        y = model.addMVar(N, vtype=GRB.BINARY, name="y")
        x = model.addMVar(N, lb=0, name="x")
        
        # u and v variables have 0 upper bound at the root node
        ub_u = np.ones(N)
        ub_u[root_idx] = 0.0
        u = model.addMVar(N, vtype=GRB.BINARY, ub=ub_u, name="u")
        
        ub_v = np.ones(N)
        ub_v[root_idx] = 0.0
        v = model.addMVar(N, vtype=GRB.BINARY, ub=ub_v, name="v")
        
        # Generation Limits
        model.addConstr(x - C_upper * y <= 0, name="gen_up")
        model.addConstr(-x + C_lower * y <= 0, name="gen_low")
        
        # Startup and Shutdown Definitions
        model.addConstr(u - v == M_diff @ y, name="start_shut")
        model.addConstr(u + v <= 1, name="u_v_mutex")
        
        # Ramping Up
        rhs_ramp_up = C_upper * ones_vec + x0_vec + (V_plus - C_upper) * y0_vec
        model.addConstr((I - P) @ x - (V_plus - C_upper) * (P @ y) <= rhs_ramp_up, name="ramp_up")
        
        # Ramping Down
        rhs_ramp_down = C_upper * ones_vec - x0_vec
        model.addConstr((P - I) @ x - (V_minus - C_upper) * y <= rhs_ramp_down, name="ramp_down")
        
        # Min Up Time Lookback
        if L > 1:
            row_up = []
            col_up = []
            val_up = []
            for i in range(N):
                curr = i
                steps = 0
                while steps < L:
                    row_up.append(i)
                    col_up.append(curr)
                    val_up.append(1.0)
                    steps += 1
                    curr = parent_idx[curr]
                    if curr is None:
                        break
            A_up = sp.csr_matrix((val_up, (row_up, col_up)), shape=(N, N))
            model.addConstr(A_up @ u <= y, name="min_up")
            
        # Min Down Time Lookback
        if ell > 1:
            row_down = []
            col_down = []
            val_down = []
            for i in range(N):
                curr = i
                steps = 0
                while steps < ell:
                    row_down.append(i)
                    col_down.append(curr)
                    val_down.append(1.0)
                    steps += 1
                    curr = parent_idx[curr]
                    if curr is None:
                        break
            A_down = sp.csr_matrix((val_down, (row_down, col_down)), shape=(N, N))
            model.addConstr(A_down @ v <= 1 - y, name="min_down")

        # Set Objective with exact vectorized computations
        obj_x = prob * (price - b)
        obj_y = -prob * a
        obj_u = -prob * U_start
        obj_u[root_idx] = 0.0
        obj_v = -prob * U_shut
        obj_v[root_idx] = 0.0
        
        model.setObjective(obj_x @ x + obj_y @ y + obj_u @ u + obj_v @ v, GRB.MAXIMIZE)
        
        # Logging Callback
        def callback(mdl, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                    x_val = mdl.cbGetSolution(x)
                    y_val = mdl.cbGetSolution(y)
                    u_val = mdl.cbGetSolution(u)
                    
                    x_nz = np.where(x_val > 1e-5)[0]
                    sol_x = {str(node_ids[idx]): float(x_val[idx]) for idx in x_nz}

                    y_nz = np.where(y_val > 0.5)[0]
                    sol_y = {str(node_ids[idx]): int(round(y_val[idx])) for idx in y_nz}

                    u_nz = np.where(u_val > 0.5)[0]
                    sol_u = {str(node_ids[idx]): int(round(u_val[idx])) for idx in u_nz if idx != root_idx}
                            
                    solution = {
                        "objective_value": obj,
                        "x": sol_x,
                        "y": sol_y,
                        "u": sol_u,
                        "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
                    }
                    if logger:
                        logger.log_solution(obj, solution)
                except Exception:
                    pass

        # Optimize
        if logger:
            model.optimize(callback)
        else:
            model.optimize()
            
        # Post-Processing
        if hasattr(model, 'SolCount') and model.SolCount > 0:
            x_val = x.X
            y_val = y.X
            u_val = u.X
            
            x_nz = np.where(x_val > 1e-5)[0]
            sol_x = {str(node_ids[idx]): float(x_val[idx]) for idx in x_nz}

            y_nz = np.where(y_val > 0.5)[0]
            sol_y = {str(node_ids[idx]): int(round(y_val[idx])) for idx in y_nz}

            u_nz = np.where(u_val > 0.5)[0]
            sol_u = {str(node_ids[idx]): int(round(u_val[idx])) for idx in u_nz if idx != root_idx}
                    
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
                "x": {}, "y": {}, "u": {},
                "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
            }

        with open(args.solution_path, 'w') as f:
            json.dump(final_solution, f, indent=2)

    except Exception as e:
        print(f"Global exception caught: {e}")
        traceback.print_exc()
        try:
            with open(args.solution_path, 'w') as f:
                json.dump({
                    "objective_value": 0.0,
                    "x": {},
                    "y": {},
                    "u": {},
                    "sparse_defaults": {"x": 0.0, "y": 0, "u": 0}
                }, f, indent=2)
        except:
            pass

if __name__ == "__main__":
    solve()