import json
import argparse
import time
from gurobipy import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n_gen = instance['n_generators']
    n_scen = instance['n_scenarios']
    T = instance['T']
    generators = instance['generators']
    demand = instance['demand_scenarios']
    reserve = instance['reserve_scenarios']
    
    # Create model
    model = Model("stochastic_uc")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 1)
    
    # Decision variables
    x = {}
    y = {}
    y_bar = {}
    q = {}
    
    for g in range(n_gen):
        for t in range(T):
            x[g, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{g}_{t}")
            y[g, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{g}_{t}")
            y_bar[g, t] = model.addVar(vtype=GRB.BINARY, name=f"ybar_{g}_{t}")
            q[g, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"q_{g}_{t}")
    
    p = {}
    p_bar = {}
    
    for g in range(n_gen):
        for s in range(n_scen):
            for t in range(T):
                p[g, s, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{g}_{s}_{t}")
                p_bar[g, s, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"pbar_{g}_{s}_{t}")
    
    model.update()
    
    # Objective
    obj = LinExpr()
    for g in range(n_gen):
        for t in range(T):
            obj += generators[g]['c_f'] * x[g, t] + q[g, t]
    for g in range(n_gen):
        for s in range(n_scen):
            for t in range(T):
                obj += (1.0 / n_scen) * generators[g]['c_g'] * p[g, s, t]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints for each generator
    for g in range(n_gen):
        gen = generators[g]
        
        for t in range(T):
            # Logic constraint: y[t] - y_bar[t] = x[t] - x[t-1]
            if t == 0:
                model.addConstr(y[g, t] - y_bar[g, t] == x[g, t], name=f"logic_{g}_{t}")
            else:
                model.addConstr(y[g, t] - y_bar[g, t] == x[g, t] - x[g, t-1], name=f"logic_{g}_{t}")
            
            # Minimum up-time
            L = gen['L']
            start_idx = max(0, t - L + 1)
            model.addConstr(
                quicksum(y[g, tau] for tau in range(start_idx, t + 1)) <= x[g, t],
                name=f"min_up_{g}_{t}"
            )
            
            # Minimum down-time
            ell = gen['ell']
            start_idx = max(0, t - ell + 1)
            model.addConstr(
                quicksum(y_bar[g, tau] for tau in range(start_idx, t + 1)) <= 1 - x[g, t],
                name=f"min_down_{g}_{t}"
            )
            
            # Tiered start-up costs
            for k in range(1, t + 1):
                sum_x = quicksum(x[g, tau] for tau in range(t - k, t))
                model.addConstr(
                    q[g, t] >= gen['startup_costs_K'][k-1] * (x[g, t] - sum_x),
                    name=f"startup_cost_{g}_{t}_{k}"
                )
            
            # Production and capacity bounds per scenario
            for s in range(n_scen):
                model.addConstr(p[g, s, t] >= gen['m'] * x[g, t], name=f"p_lb_{g}_{s}_{t}")
                model.addConstr(p[g, s, t] <= p_bar[g, s, t], name=f"p_ub_{g}_{s}_{t}")
                model.addConstr(p_bar[g, s, t] <= gen['M'] * x[g, t], name=f"pbar_ub_{g}_{s}_{t}")
                
                if t > 0:
                    # Ramp up
                    model.addConstr(
                        p[g, s, t] - p[g, s, t-1] <= gen['RU'] * x[g, t-1] + gen['SU'] * y[g, t],
                        name=f"ramp_up_{g}_{s}_{t}"
                    )
                    # Ramp down
                    model.addConstr(
                        p[g, s, t-1] - p[g, s, t] <= gen['RD'] * x[g, t] + gen['SD'] * y_bar[g, t],
                        name=f"ramp_down_{g}_{s}_{t}"
                    )
    
    # Demand and reserve constraints
    for s in range(n_scen):
        for t in range(T):
            model.addConstr(
                quicksum(p[g, s, t] for g in range(n_gen)) >= demand[s][t],
                name=f"demand_{s}_{t}"
            )
            model.addConstr(
                quicksum(p_bar[g, s, t] for g in range(n_gen)) >= demand[s][t] + reserve[s][t],
                name=f"reserve_{s}_{t}"
            )
    
    model.update()
    
    # Callback for logging incumbent solutions
    best_logged_obj = float('inf')
    
    def callback(model, where):
        nonlocal best_logged_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_logged_obj - 1e-6:
                best_logged_obj = obj_val
                if logger:
                    x_sol = [[int(round(model.cbGetSolution(x[g, t]))) for t in range(T)] for g in range(n_gen)]
                    y_sol = [[int(round(model.cbGetSolution(y[g, t]))) for t in range(T)] for g in range(n_gen)]
                    ybar_sol = [[int(round(model.cbGetSolution(y_bar[g, t]))) for t in range(T)] for g in range(n_gen)]
                    q_sol = [[model.cbGetSolution(q[g, t]) for t in range(T)] for g in range(n_gen)]
                    p_sol = [[[model.cbGetSolution(p[g, s, t]) for t in range(T)] for s in range(n_scen)] for g in range(n_gen)]
                    pbar_sol = [[[model.cbGetSolution(p_bar[g, s, t]) for t in range(T)] for s in range(n_scen)] for g in range(n_gen)]
                    
                    solution = {
                        "objective_value": obj_val,
                        "x": x_sol,
                        "y": y_sol,
                        "y_bar": ybar_sol,
                        "q": q_sol,
                        "p": p_sol,
                        "p_bar": pbar_sol
                    }
                    logger.log_solution(obj_val, solution)
    
    model.optimize(callback)
    
    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        x_sol = [[int(round(x[g, t].X)) for t in range(T)] for g in range(n_gen)]
        y_sol = [[int(round(y[g, t].X)) for t in range(T)] for g in range(n_gen)]
        ybar_sol = [[int(round(y_bar[g, t].X)) for t in range(T)] for g in range(n_gen)]
        q_sol = [[q[g, t].X for t in range(T)] for g in range(n_gen)]
        p_sol = [[[p[g, s, t].X for t in range(T)] for s in range(n_scen)] for g in range(n_gen)]
        pbar_sol = [[[p_bar[g, s, t].X for t in range(T)] for s in range(n_scen)] for g in range(n_gen)]
        
        solution = {
            "objective_value": obj_val,
            "x": x_sol,
            "y": y_sol,
            "y_bar": ybar_sol,
            "q": q_sol,
            "p": p_sol,
            "p_bar": pbar_sol
        }
        
        if logger and obj_val < best_logged_obj - 1e-6:
            logger.log_solution(obj_val, solution)
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # No feasible solution found
        solution = {
            "objective_value": float('inf'),
            "x": [[0]*T for _ in range(n_gen)],
            "y": [[0]*T for _ in range(n_gen)],
            "y_bar": [[0]*T for _ in range(n_gen)],
            "q": [[0.0]*T for _ in range(n_gen)],
            "p": [[[0.0]*T for _ in range(n_scen)] for _ in range(n_gen)],
            "p_bar": [[[0.0]*T for _ in range(n_scen)] for _ in range(n_gen)]
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)

if __name__ == "__main__":
    main()