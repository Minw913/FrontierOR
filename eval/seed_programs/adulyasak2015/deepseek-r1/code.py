import argparse
import json
import gurobipy as gp
from solution_logger import SolutionLogger

def callback_function(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        if hasattr(model, '_logger') and model._logger:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            model._logger.log(obj)

def main(instance_path, solution_path, time_limit, log_path):
    # Initialize logger if log_path provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # Extract instance parameters
    n = instance['n']
    T = instance['T']
    m = instance['m']
    Q = instance['Q']
    C = instance['C']
    f_cost = instance['f']
    u_cost = instance['u']
    transportation_costs = instance['transportation_costs']
    h = instance['h']
    L = instance['L']
    I0 = instance['I0']
    sigma = instance['sigma']
    n_scenarios = instance['n_scenarios']
    scenario_probabilities = instance['scenario_probabilities']
    demand_scenarios = instance['demand_scenarios']
    
    # Create model
    model = gp.Model('ProductionRouting')
    
    # Store logger in model for callback access
    model._logger = logger
    
    # Stage 1 variables
    y = {}  # Setup indicators
    for t in range(1, T+1):
        y[t] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{t}")
    
    z = {}  # Visit indicators
    x = {}  # Edge traversals
    for t in range(1, T+1):
        for k in range(1, m+1):
            for i in range(n+1):
                z[i, k, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}_{k}_{t}")
            for i in range(n+1):
                for j in range(n+1):
                    if i != j:
                        x[i, j, k, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")
    
    # Stage 2 variables
    p = {}  # Production quantities
    I = {}  # Inventory levels
    e = {}  # Unmet demand
    q = {}  # Delivery quantities
    for omega in range(n_scenarios):
        for t in range(1, T+1):
            p[t, omega] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"p_{t}_{omega}")
            for i in range(n+1):
                I[i, t, omega] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"I_{i}_{t}_{omega}")
            for i in range(1, n+1):
                e[i, t, omega] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"e_{i}_{t}_{omega}")
            for k in range(1, m+1):
                for i in range(1, n+1):
                    q[i, k, t, omega] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"q_{i}_{k}_{t}_{omega}")
    
    # Constraints
    # Each customer visited by at most one vehicle per period
    for i in range(1, n+1):
        for t in range(1, T+1):
            model.addConstr(
                gp.quicksum(z[i, k, t] for k in range(1, m+1)) <= 1,
                f"visit_once_{i}_{t}"
            )
    
    # Routing constraints
    for t in range(1, T+1):
        for k in range(1, m+1):
            # Plant flow conservation
            model.addConstr(
                gp.quicksum(x[0, j, k, t] for j in range(1, n+1)) == z[0, k, t],
                f"plant_out_{k}_{t}"
            )
            model.addConstr(
                gp.quicksum(x[j, 0, k, t] for j in range(1, n+1)) == z[0, k, t],
                f"plant_in_{k}_{t}"
            )
            
            # Customer flow conservation
            for i in range(1, n+1):
                model.addConstr(
                    gp.quicksum(x[j, i, k, t] for j in range(n+1) if j != i) == z[i, k, t],
                    f"in_flow_{i}_{k}_{t}"
                )
                model.addConstr(
                    gp.quicksum(x[i, j, k, t] for j in range(n+1) if j != i) == z[i, k, t],
                    f"out_flow_{i}_{k}_{t}"
                )
            
            # MTZ subtour elimination
            u = {}
            for i in range(1, n+1):
                u[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=n, name=f"u_{i}_{k}_{t}")
            for i in range(1, n+1):
                for j in range(1, n+1):
                    if i != j:
                        model.addConstr(
                            u[i] - u[j] + n * x[i, j, k, t] <= n-1,
                            f"mtz_{i}_{j}_{k}_{t}"
                        )
    
    # Stage 2 constraints
    for omega in range(n_scenarios):
        for t in range(1, T+1):
            # Plant inventory balance
            total_shipment = gp.quicksum(
                q[i, k, t, omega] 
                for i in range(1, n+1) 
                for k in range(1, m+1)
            )
            prev_inv_plant = I0[0] if t == 1 else I[0, t-1, omega]
            model.addConstr(
                prev_inv_plant + p[t, omega] == I[0, t, omega] + total_shipment,
                f"plant_balance_{t}_{omega}"
            )
            
            # Production constraints
            model.addConstr(p[t, omega] <= C * y[t], f"prod_setup_{t}_{omega}")
            model.addConstr(p[t, omega] <= C, f"prod_cap_{t}_{omega}")
            
            # Customer constraints
            for i in range(1, n+1):
                d_it = demand_scenarios[omega][i-1][t-1]
                total_delivery = gp.quicksum(
                    q[i, k, t, omega] for k in range(1, m+1)
                )
                prev_inv_cust = I0[i] if t == 1 else I[i, t-1, omega]
                model.addConstr(
                    prev_inv_cust + total_delivery + e[i, t, omega] == d_it + I[i, t, omega],
                    f"cust_balance_{i}_{t}_{omega}"
                )
                model.addConstr(
                    I[i, t, omega] + d_it <= L[i],
                    f"storage_{i}_{t}_{omega}"
                )
                model.addConstr(
                    I[i, t, omega] <= L[i],
                    f"end_inv_{i}_{t}_{omega}"
                )
            
            # Vehicle capacity constraints
            for k in range(1, m+1):
                model.addConstr(
                    gp.quicksum(q[i, k, t, omega] for i in range(1, n+1)) <= Q,
                    f"vehicle_cap_{k}_{t}_{omega}"
                )
            
            # Delivery constraints
            for k in range(1, m+1):
                for i in range(1, n+1):
                    model.addConstr(
                        q[i, k, t, omega] <= L[i] * z[i, k, t],
                        f"delivery_bound_{i}_{k}_{t}_{omega}"
                    )
    
    # Objective function
    setup_cost = gp.quicksum(f_cost * y[t] for t in range(1, T+1))
    
    transportation_cost = gp.quicksum(
        transportation_costs[i][j] * x[i, j, k, t]
        for i in range(n+1) for j in range(n+1) if i != j
        for k in range(1, m+1) for t in range(1, T+1)
    )
    
    expected_production_cost = gp.quicksum(
        scenario_probabilities[omega] * u_cost * p[t, omega]
        for t in range(1, T+1) for omega in range(n_scenarios)
    )
    
    expected_holding_cost = gp.quicksum(
        scenario_probabilities[omega] * h[i] * I[i, t, omega]
        for i in range(n+1) for t in range(1, T+1) for omega in range(n_scenarios)
    )
    
    expected_penalty_cost = gp.quicksum(
        scenario_probabilities[omega] * sigma[i-1] * e[i, t, omega]
        for i in range(1, n+1) for t in range(1, T+1) for omega in range(n_scenarios)
    )
    
    total_cost = (
        setup_cost + 
        transportation_cost + 
        expected_production_cost + 
        expected_holding_cost + 
        expected_penalty_cost
    )
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    
    # Solver parameters
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Optimize with callback
    model.optimize(callback_function)
    
    # Prepare solution dictionary
    solution = {
        "objective_value": None,
        "y": {},
        "z": {},
        "x": {},
        "model_variables": {}
    }
    
    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        
        # Stage 1 variables
        for t in range(1, T+1):
            solution["y"][str(t)] = int(round(y[t].X))
        
        for t in range(1, T+1):
            for k in range(1, m+1):
                for i in range(n+1):
                    if abs(z[i, k, t].X) > 0.5:
                        solution["z"][f"{i}_{k}_{t}"] = 1
        
        for t in range(1, T+1):
            for k in range(1, m+1):
                for i in range(n+1):
                    for j in range(n+1):
                        if i != j and abs(x[i, j, k, t].X) > 0.5:
                            solution["x"][f"{i}_{j}_{k}_{t}"] = 1
        
        # Stage 2 variables
        for omega in range(n_scenarios):
            for t in range(1, T+1):
                if abs(p[t, omega].X) > 1e-6:
                    solution["model_variables"][f"p_{t}_{omega}"] = p[t, omega].X
                for i in range(n+1):
                    if abs(I[i, t, omega].X) > 1e-6:
                        solution["model_variables"][f"I_{i}_{t}_{omega}"] = I[i, t, omega].X
                for i in range(1, n+1):
                    if abs(e[i, t, omega].X) > 1e-6:
                        solution["model_variables"][f"e_{i}_{t}_{omega}"] = e[i, t, omega].X
                    for k in range(1, m+1):
                        if abs(q[i, k, t, omega].X) > 1e-6:
                            solution["model_variables"][f"q_{i}_{k}_{t}_{omega}"] = q[i, k, t, omega].X
        
        # Log final solution if logger exists
        if logger:
            logger.log_solution(model.ObjVal, solution)
    
    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    main(
        args.instance_path,
        args.solution_path,
        args.time_limit,
        args.log_path
    )