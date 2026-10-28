import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n_generators = data['n_generators']
    n_scenarios = data['n_scenarios']
    T = data['T']
    generators = data['generators']
    demand_scenarios = data['demand_scenarios']
    reserve_scenarios = data['reserve_scenarios']
    
    # Create model
    model = gp.Model("unit_commitment")
    model.setParam('TimeLimit', max(1, time_limit - 5))  # Reserve 5 seconds for output
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-6)
    
    # Index sets
    G = range(n_generators)
    S = range(n_scenarios)
    Periods = range(T)
    
    # Decision variables
    # Commitment, startup, shutdown (scenario-independent)
    x = {}  # x[g,t] = 1 if generator g is on in period t
    y = {}  # y[g,t] = 1 if generator g starts up in period t
    y_bar = {}  # y_bar[g,t] = 1 if generator g shuts down in period t
    q = {}  # q[g,t] = startup cost for generator g in period t
    
    for g in G:
        for t in Periods:
            x[g, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{g}_{t}")
            y[g, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{g}_{t}")
            y_bar[g, t] = model.addVar(vtype=GRB.BINARY, name=f"ybar_{g}_{t}")
            q[g, t] = model.addVar(lb=0, name=f"q_{g}_{t}")
    
    # Production and availability (scenario-dependent)
    p = {}  # p[g,s,t] = production of generator g in scenario s, period t
    p_bar = {}  # p_bar[g,s,t] = available capacity
    
    for g in G:
        gen = generators[g]
        for s in S:
            for t in Periods:
                p[g, s, t] = model.addVar(lb=0, name=f"p_{g}_{s}_{t}")
                p_bar[g, s, t] = model.addVar(lb=0, name=f"pbar_{g}_{s}_{t}")
    
    model.update()
    
    # Constraints
    
    # 1. Logical linking: y[g,t] - y_bar[g,t] = x[g,t] - x[g,t-1]
    # All generators are offline at start (x[g,-1] = 0)
    for g in G:
        for t in Periods:
            if t == 0:
                model.addConstr(y[g, t] - y_bar[g, t] == x[g, t], name=f"logic_{g}_{t}")
            else:
                model.addConstr(y[g, t] - y_bar[g, t] == x[g, t] - x[g, t - 1], name=f"logic_{g}_{t}")
    
    # 2. Minimum up-time constraints
    for g in G:
        gen = generators[g]
        L = gen['L']
        for t in Periods:
            # Sum of startups in window [t-L+1, t] <= x[g,t]
            start_idx = max(0, t - L + 1)
            model.addConstr(
                gp.quicksum(y[g, tau] for tau in range(start_idx, t + 1)) <= x[g, t],
                name=f"minup_{g}_{t}"
            )
    
    # 3. Minimum down-time constraints
    for g in G:
        gen = generators[g]
        ell = gen['ell']
        for t in Periods:
            start_idx = max(0, t - ell + 1)
            model.addConstr(
                gp.quicksum(y_bar[g, tau] for tau in range(start_idx, t + 1)) <= 1 - x[g, t],
                name=f"mindn_{g}_{t}"
            )
    
    # 4. Startup cost constraints
    # q[g,t] >= startup_costs_K[k-1] * (x[g,t] - sum_{j=t-k}^{t-1} x[g,j])
    # for k = 1 to t (since periods before 0 have x=0)
    for g in G:
        gen = generators[g]
        startup_costs = gen['startup_costs_K']
        for t in Periods:
            # k ranges from 1 to t+1 (since we have t periods before t: 0..t-1, plus initial off)
            # But also limited by T
            max_k = t + 1  # at most t+1 consecutive inactive periods possible (periods 0..t-1 plus initial)
            # But startup_costs has length T, indexed 0..T-1 for k=1..T
            max_k = min(max_k, T)
            for k in range(1, max_k + 1):
                # Sum of x[g,j] for j = t-k to t-1
                sum_x_prev = gp.quicksum(
                    x[g, j] for j in range(max(0, t - k), t)
                )
                # Note: periods before 0 are all off (x=0), so if t-k < 0, those terms are 0
                model.addConstr(
                    q[g, t] >= startup_costs[k - 1] * (x[g, t] - sum_x_prev),
                    name=f"scost_{g}_{t}_{k}"
                )
    
    # 5. Production bounds
    for g in G:
        gen = generators[g]
        for s in S:
            for t in Periods:
                # When off, production = 0: p[g,s,t] <= M * x[g,t]
                model.addConstr(p[g, s, t] <= gen['M'] * x[g, t], name=f"pmax_{g}_{s}_{t}")
                # When on, production >= m: p[g,s,t] >= m * x[g,t]
                model.addConstr(p[g, s, t] >= gen['m'] * x[g, t], name=f"pmin_{g}_{s}_{t}")
                # p[g,s,t] <= p_bar[g,s,t]
                model.addConstr(p[g, s, t] <= p_bar[g, s, t], name=f"pavail_{g}_{s}_{t}")
                # p_bar[g,s,t] <= M * x[g,t]
                model.addConstr(p_bar[g, s, t] <= gen['M'] * x[g, t], name=f"pbarmax_{g}_{s}_{t}")
                # p_bar when off = 0
                # Already covered by p_bar <= M * x[g,t]
    
    # 6. Ramp-up constraints
    # p[g,s,t] - p[g,s,t-1] <= RU * x[g,t-1] + SU * y[g,t]
    # For t=0: p[g,s,0] - 0 <= RU * 0 + SU * y[g,0] = SU * y[g,0]
    # But since all generators start offline, x[g,-1]=0
    for g in G:
        gen = generators[g]
        RU = gen['RU']
        SU_val = gen['SU']
        for s in S:
            for t in Periods:
                if t == 0:
                    model.addConstr(
                        p[g, s, t] <= SU_val * y[g, t],
                        name=f"rampup_{g}_{s}_{t}"
                    )
                else:
                    model.addConstr(
                        p[g, s, t] - p[g, s, t - 1] <= RU * x[g, t - 1] + SU_val * y[g, t],
                        name=f"rampup_{g}_{s}_{t}"
                    )
    
    # 7. Ramp-down constraints
    # p[g,s,t-1] - p[g,s,t] <= RD * x[g,t] + SD * y_bar[g,t]
    for g in G:
        gen = generators[g]
        RD = gen['RD']
        SD_val = gen['SD']
        for s in S:
            for t in Periods:
                if t == 0:
                    # p[g,s,-1] = 0 (offline), so 0 - p[g,s,0] <= RD * x[g,0] + SD * y_bar[g,0]
                    # This is automatically satisfied since p >= 0 and RD, SD >= 0
                    model.addConstr(
                        -p[g, s, t] <= RD * x[g, t] + SD_val * y_bar[g, t],
                        name=f"rampdn_{g}_{s}_{t}"
                    )
                else:
                    model.addConstr(
                        p[g, s, t - 1] - p[g, s, t] <= RD * x[g, t] + SD_val * y_bar[g, t],
                        name=f"rampdn_{g}_{s}_{t}"
                    )
    
    # 8. Demand satisfaction
    for s in S:
        for t in Periods:
            model.addConstr(
                gp.quicksum(p[g, s, t] for g in G) >= demand_scenarios[s][t],
                name=f"demand_{s}_{t}"
            )
    
    # 9. Reserve requirement
    for s in S:
        for t in Periods:
            model.addConstr(
                gp.quicksum(p_bar[g, s, t] for g in G) >= demand_scenarios[s][t] + reserve_scenarios[s][t],
                name=f"reserve_{s}_{t}"
            )
    
    # Objective function
    # Fixed + startup costs (scenario-independent)
    fixed_cost = gp.quicksum(
        generators[g]['c_f'] * x[g, t] + q[g, t]
        for g in G for t in Periods
    )
    
    # Production cost (averaged over scenarios)
    prob = 1.0 / n_scenarios
    prod_cost = prob * gp.quicksum(
        generators[g]['c_g'] * p[g, s, t]
        for g in G for s in S for t in Periods
    )
    
    model.setObjective(fixed_cost + prod_cost, GRB.MINIMIZE)
    
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
        
        x_sol = [[int(round(x[g, t].X)) for t in Periods] for g in G]
        y_sol = [[int(round(y[g, t].X)) for t in Periods] for g in G]
        ybar_sol = [[int(round(y_bar[g, t].X)) for t in Periods] for g in G]
        q_sol = [[q[g, t].X for t in Periods] for g in G]
        p_sol = [[[p[g, s, t].X for t in Periods] for s in S] for g in G]
        pbar_sol = [[[p_bar[g, s, t].X for t in Periods] for s in S] for g in G]
        
        solution = {
            "objective_value": obj_val,
            "x": x_sol,
            "y": y_sol,
            "y_bar": ybar_sol,
            "q": q_sol,
            "p": p_sol,
            "p_bar": pbar_sol
        }
    else:
        # No feasible solution found - return trivial
        solution = {
            "objective_value": float('inf'),
            "x": [[0] * T for _ in G],
            "y": [[0] * T for _ in G],
            "y_bar": [[0] * T for _ in G],
            "q": [[0.0] * T for _ in G],
            "p": [[[0.0] * T for _ in S] for _ in G],
            "p_bar": [[[0.0] * T for _ in S] for _ in G]
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