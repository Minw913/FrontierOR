import gurobipy as gp
from gurobipy import GRB, quicksum
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
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n_generators = data['n_generators']
    n_scenarios = data['n_scenarios']
    T = data['T']
    generators = data['generators']
    demand_scenarios = data['demand_scenarios']
    reserve_scenarios = data['reserve_scenarios']
    
    model = gp.Model("thermal_generation_planning")
    model.setParam('OutputFlag', 1)
    model.setParam('Threads', 1)
    
    x = []
    y = []
    y_bar = []
    q = []
    for i in range(n_generators):
        x_i = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}") for t in range(T)]
        y_i = [model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}") for t in range(T)]
        y_bar_i = [model.addVar(vtype=GRB.BINARY, name=f"y_bar_{i}_{t}") for t in range(T)]
        q_i = [model.addVar(lb=0, name=f"q_{i}_{t}") for t in range(T)]
        x.append(x_i)
        y.append(y_i)
        y_bar.append(y_bar_i)
        q.append(q_i)
    
    p = []
    p_bar = []
    for i in range(n_generators):
        p_i = []
        p_bar_i = []
        for s in range(n_scenarios):
            p_i_s = [model.addVar(lb=0, name=f"p_{i}_{s}_{t}") for t in range(T)]
            p_bar_i_s = [model.addVar(lb=0, name=f"p_bar_{i}_{s}_{t}") for t in range(T)]
            p_i.append(p_i_s)
            p_bar_i.append(p_bar_i_s)
        p.append(p_i)
        p_bar.append(p_bar_i)
    
    for i in range(n_generators):
        gen = generators[i]
        L_i = gen['L']
        ell_i = gen['ell']
        startup_costs = gen['startup_costs_K']
        
        for t in range(T):
            if t == 0:
                model.addConstr(y[i][t] - y_bar[i][t] == x[i][t], f"state_trans_{i}_{t}")
            else:
                model.addConstr(y[i][t] - y_bar[i][t] == x[i][t] - x[i][t-1], f"state_trans_{i}_{t}")
            
            for k in range(t, min(t + L_i, T)):
                model.addConstr(x[i][k] >= y[i][t], f"min_up_{i}_{t}_{k}")
            
            for k in range(t, min(t + ell_i, T)):
                model.addConstr(1 - x[i][k] >= y_bar[i][t], f"min_down_{i}_{t}_{k}")
            
            if t >= 1:
                for k in range(1, min(t, T) + 1):
                    s_expr = x[i][t]
                    for j in range(1, k+1):
                        s_expr -= x[i][t - j]
                    model.addConstr(q[i][t] >= startup_costs[k-1] * s_expr, f"startup_cost_{i}_{t}_k{k}")
            
            model.addConstr(q[i][t] <= startup_costs[-1] * y[i][t], f"q_upper_{i}_{t}")
    
    for s in range(n_scenarios):
        for i in range(n_generators):
            gen = generators[i]
            M_i = gen['M']
            m_i = gen['m']
            RU_i = gen['RU']
            SU_i = gen['SU']
            RD_i = gen['RD']
            SD_i = gen['SD']
            
            for t in range(T):
                model.addConstr(p[i][s][t] <= p_bar[i][s][t], f"p_le_pbar_{i}_{s}_{t}")
                model.addConstr(p_bar[i][s][t] <= M_i * x[i][t], f"pbar_ub_{i}_{s}_{t}")
                model.addConstr(p[i][s][t] <= M_i * x[i][t], f"p_ub_{i}_{s}_{t}")
                model.addConstr(p[i][s][t] >= m_i * x[i][t], f"p_lb_{i}_{s}_{t}")
                
                if t >= 1:
                    model.addConstr(
                        p[i][s][t] - p[i][s][t-1] <= RU_i * x[i][t-1] + SU_i * y[i][t],
                        f"ramp_up_{i}_{s}_{t}"
                    )
                    model.addConstr(
                        p[i][s][t-1] - p[i][s][t] <= RD_i * x[i][t] + SD_i * y_bar[i][t],
                        f"ramp_down_{i}_{s}_{t}"
                    )
        
        for t in range(T):
            total_p = quicksum(p[i][s][t] for i in range(n_generators))
            total_p_bar = quicksum(p_bar[i][s][t] for i in range(n_generators))
            model.addConstr(total_p >= demand_scenarios[s][t], f"demand_{s}_{t}")
            model.addConstr(total_p_bar >= demand_scenarios[s][t] + reserve_scenarios[s][t], f"reserve_{s}_{t}")
    
    fixed_cost = quicksum(gen['c_f'] * x[i][t] for i, gen in enumerate(generators) for t in range(T))
    startup_cost = quicksum(q[i][t] for i in range(n_generators) for t in range(T))
    production_cost = (1.0 / n_scenarios) * quicksum(
        gen['c_g'] * p[i][s][t] 
        for s in range(n_scenarios)
        for i, gen in enumerate(generators)
        for t in range(T)
    )
    total_cost = fixed_cost + startup_cost + production_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    model.setParam('TimeLimit', args.time_limit)
    
    if logger:
        model._logger = logger
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                model._logger.log(obj)
        model.optimize(callback)
    else:
        model.optimize()
    
    if model.SolCount > 0:
        sol_x = [[int(round(x[i][t].X)) for t in range(T)] for i in range(n_generators)]
        sol_y = [[int(round(y[i][t].X)) for t in range(T)] for i in range(n_generators)]
        sol_y_bar = [[int(round(y_bar[i][t].X)) for t in range(T)] for i in range(n_generators)]
        sol_q = [[q[i][t].X for t in range(T)] for i in range(n_generators)]
        sol_p = [[[p[i][s][t].X for t in range(T)] for s in range(n_scenarios)] for i in range(n_generators)]
        sol_p_bar = [[[p_bar[i][s][t].X for t in range(T)] for s in range(n_scenarios)] for i in range(n_generators)]
        objective_value = model.ObjVal
    else:
        sol_x = [[0] * T for _ in range(n_generators)]
        sol_y = [[0] * T for _ in range(n_generators)]
        sol_y_bar = [[0] * T for _ in range(n_generators)]
        sol_q = [[0.0] * T for _ in range(n_generators)]
        sol_p = [[[0.0] * T for _ in range(n_scenarios)] for _ in range(n_generators)]
        sol_p_bar = [[[0.0] * T for _ in range(n_scenarios)] for _ in range(n_generators)]
        objective_value = 1e20
    
    solution = {
        "objective_value": objective_value,
        "x": sol_x,
        "y": sol_y,
        "y_bar": sol_y_bar,
        "q": sol_q,
        "p": sol_p,
        "p_bar": sol_p_bar
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()