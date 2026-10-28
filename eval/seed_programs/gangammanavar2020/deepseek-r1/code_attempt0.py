import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def solve_extensive_form(instance, total_constant, time_limit, logger):
    N = instance["N"]
    second_stage = instance["second_stage"]
    holding_costs = second_stage["holding_costs"]
    penalty_costs = second_stage["penalty_costs"]
    transshipment_costs_mean = second_stage["transshipment_costs_mean"]
    demand_distribution = second_stage["demand_distribution"]
    scenarios_demand = demand_distribution["scenarios"]
    probabilities = demand_distribution["probabilities"]
    num_scenarios = demand_distribution["num_scenarios"]
    s_lb = instance["first_stage"]["variable_lower_bounds"]
    
    model = gp.Model("ExtensiveForm")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    if time_limit > 0:
        model.setParam('TimeLimit', time_limit)
    
    s = model.addVars(N, lb=s_lb, name="s")
    f = model.addVars(num_scenarios, N, lb=0.0, name="f")
    t = model.addVars(num_scenarios, N, N, lb=0.0, name="t")
    
    obj = gp.LinExpr()
    obj.add(gp.quicksum(holding_costs[i] * s[i] for i in range(N)))
    
    for k in range(num_scenarios):
        prob = probabilities[k]
        for i in range(N):
            obj.add(prob * (- (holding_costs[i] + penalty_costs[i])) * f[k, i])
        for i in range(N):
            for j in range(N):
                if i != j:
                    coeff = transshipment_costs_mean[i][j] - holding_costs[i] - penalty_costs[j]
                    obj.add(prob * coeff * t[k, i, j])
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    for k in range(num_scenarios):
        demand_k = scenarios_demand[k]
        for i in range(N):
            model.addConstr(
                f[k, i] + gp.quicksum(t[k, i, j] for j in range(N) if j != i) <= s[i],
                f"flow_balance_out_{k}_{i}"
            )
        for i in range(N):
            model.addConstr(
                f[k, i] + gp.quicksum(t[k, j, i] for j in range(N) if j != i) <= demand_k[i],
                f"flow_balance_in_{k}_{i}"
            )
    
    model.optimize()
    
    if model.SolCount == 0:
        raise RuntimeError("No solution found in extensive form")
    
    s_val = [s[i].X for i in range(N)]
    total_obj = model.ObjVal + total_constant
    
    if logger:
        logger.log(total_obj)
    
    scenario_solutions = []
    for k in range(num_scenarios):
        f_val = [f[k, i].X for i in range(N)]
        t_val = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    t_val[i][j] = t[k, i, j].X
        
        e_val = []
        r_val = []
        q_val = []
        for i in range(N):
            outflow = sum(t_val[i][j] for j in range(N) if j != i)
            e_i = s_val[i] - f_val[i] - outflow
            e_val.append(e_i)
            inflow = sum(t_val[j][i] for j in range(N) if j != i)
            r_i = scenarios_demand[k][i] - f_val[i] - inflow
            r_val.append(r_i)
            q_val.append(s_val[i] - e_i)
        
        scenario_solutions.append({
            "e": e_val,
            "f": f_val,
            "q": q_val,
            "r": r_val,
            "t": t_val
        })
    
    return total_obj, s_val, scenario_solutions

def solve_benders(instance, total_constant, time_limit, logger):
    N = instance["N"]
    second_stage = instance["second_stage"]
    holding_costs = second_stage["holding_costs"]
    penalty_costs = second_stage["penalty_costs"]
    transshipment_costs_mean = second_stage["transshipment_costs_mean"]
    demand_distribution = second_stage["demand_distribution"]
    scenarios_demand = demand_distribution["scenarios"]
    probabilities = demand_distribution["probabilities"]
    num_scenarios = demand_distribution["num_scenarios"]
    s_lb = instance["first_stage"]["variable_lower_bounds"]
    
    master = gp.Model("BendersMaster")
    master.setParam('OutputFlag', 0)
    master.setParam('Threads', 1)
    
    s = master.addVars(N, lb=s_lb, name="s")
    theta = master.addVar(lb=-GRB.INFINITY, name="theta")
    
    master.setObjective(
        gp.quicksum(holding_costs[i] * s[i] for i in range(N)) + theta,
        GRB.MINIMIZE
    )
    
    UB = float('inf')
    LB = -float('inf')
    s_best = None
    start_time = time.time()
    iter_count = 0
    
    while time.time() - start_time < time_limit:
        iter_count += 1
        master.optimize()
        if master.Status == GRB.OPTIMAL:
            LB = master.ObjVal
            s_current = [s[i].X for i in range(N)]
        else:
            break
        
        total_second_stage = 0.0
        u_sum = np.zeros(N)
        cut_constant = 0.0
        
        for k in range(num_scenarios):
            demand_k = scenarios_demand[k]
            sub_model = gp.Model("BendersSub")
            sub_model.setParam('OutputFlag', 0)
            sub_model.setParam('Threads', 1)
            
            f = sub_model.addVars(N, lb=0.0, name="f")
            t = sub_model.addVars(N, N, lb=0.0, name="t")
            
            obj_sub = gp.LinExpr()
            for i in range(N):
                obj_sub.add(- (holding_costs[i] + penalty_costs[i]) * f[i])
            for i in range(N):
                for j in range(N):
                    if i != j:
                        coeff = transshipment_costs_mean[i][j] - holding_costs[i] - penalty_costs[j]
                        obj_sub.add(coeff * t[i, j])
            
            sub_model.setObjective(obj_sub, GRB.MINIMIZE)
            
            constr_out = {}
            for i in range(N):
                constr_out[i] = sub_model.addConstr(
                    f[i] + gp.quicksum(t[i, j] for j in range(N) if j != i) <= s_current[i],
                    f"out_{i}"
                )
            for i in range(N):
                sub_model.addConstr(
                    f[i] + gp.quicksum(t[j, i] for j in range(N) if j != i) <= demand_k[i],
                    f"in_{i}"
                )
            
            sub_model.optimize()
            
            if sub_model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Subproblem {k} not optimal")
            
            Q_k = sub_model.ObjVal
            total_second_stage += probabilities[k] * Q_k
            u_k = [constr_out[i].Pi for i in range(N)]
            cut_constant += probabilities[k] * (Q_k + np.dot(u_k, s_current))
            for i in range(N):
                u_sum[i] += probabilities[k] * u_k[i]
        
        total_cost = sum(holding_costs[i] * s_current[i] for i in range(N)) + total_second_stage
        total_cost_with_constant = total_cost + total_constant
        
        if total_cost_with_constant < UB:
            UB = total_cost_with_constant
            s_best = s_current.copy()
            if logger:
                logger.log(UB)
        
        expr = gp.LinExpr()
        expr.addConstant(cut_constant)
        for i in range(N):
            expr.addTerm(-u_sum[i], s[i])
        master.addConstr(theta >= expr, f"cut_{iter_count}")
        
        if UB - LB < 1e-5:
            break
        
        if time.time() - start_time >= time_limit:
            break
    
    if s_best is None:
        raise RuntimeError("Benders did not find a feasible solution")
    
    scenario_solutions = []
    for k in range(num_scenarios):
        demand_k = scenarios_demand[k]
        sub_model = gp.Model("BendersSubFinal")
        sub_model.setParam('OutputFlag', 0)
        sub_model.setParam('Threads', 1)
        
        f = sub_model.addVars(N, lb=0.0, name="f")
        t = sub_model.addVars(N, N, lb=0.0, name="t")
        
        obj_sub = gp.LinExpr()
        for i in range(N):
            obj_sub.add(- (holding_costs[i] + penalty_costs[i]) * f[i])
        for i in range(N):
            for j in range(N):
                if i != j:
                    coeff = transshipment_costs_mean[i][j] - holding_costs[i] - penalty_costs[j]
                    obj_sub.add(coeff * t[i, j])
        
        sub_model.setObjective(obj_sub, GRB.MINIMIZE)
        
        for i in range(N):
            sub_model.addConstr(
                f[i] + gp.quicksum(t[i, j] for j in range(N) if j != i) <= s_best[i],
                f"out_{i}"
            )
        for i in range(N):
            sub_model.addConstr(
                f[i] + gp.quicksum(t[j, i] for j in range(N) if j != i) <= demand_k[i],
                f"in_{i}"
            )
        
        sub_model.optimize()
        
        if sub_model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Final subproblem {k} not optimal")
        
        f_val = [f[i].X for i in range(N)]
        t_val = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    t_val[i][j] = t[i, j].X
        
        e_val = []
        r_val = []
        q_val = []
        for i in range(N):
            outflow = sum(t_val[i][j] for j in range(N) if j != i)
            e_i = s_best[i] - f_val[i] - outflow
            e_val.append(e_i)
            inflow = sum(t_val[j][i] for j in range(N) if j != i)
            r_i = demand_k[i] - f_val[i] - inflow
            r_val.append(r_i)
            q_val.append(s_best[i] - e_i)
        
        scenario_solutions.append({
            "e": e_val,
            "f": f_val,
            "q": q_val,
            "r": r_val,
            "t": t_val
        })
    
    return UB, s_best, scenario_solutions

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
    
    second_stage = instance["second_stage"]
    penalty_costs = second_stage["penalty_costs"]
    demand_distribution = second_stage["demand_distribution"]
    scenarios_demand = demand_distribution["scenarios"]
    probabilities = demand_distribution["probabilities"]
    num_scenarios = demand_distribution["num_scenarios"]
    
    total_constant = 0.0
    for k in range(num_scenarios):
        prob = probabilities[k]
        for i in range(len(penalty_costs)):
            total_constant += prob * penalty_costs[i] * scenarios_demand[k][i]
    
    N = instance["N"]
    S = num_scenarios
    total_vars = N + S * (N + N * (N - 1))
    
    start_time = time.time()
    time_left = args.time_limit - (time.time() - start_time)
    
    if total_vars <= 1000000 and time_left > 0:
        try:
            total_obj, s_val, scenario_solutions = solve_extensive_form(
                instance, total_constant, time_left, logger
            )
        except Exception as e:
            print(f"Extensive form failed: {e}, falling back to Benders")
            time_left = args.time_limit - (time.time() - start_time)
            if time_left > 0:
                total_obj, s_val, scenario_solutions = solve_benders(
                    instance, total_constant, time_left, logger
                )
            else:
                raise RuntimeError("Time limit exceeded before solving")
    else:
        total_obj, s_val, scenario_solutions = solve_benders(
            instance, total_constant, time_left, logger
        )
    
    solution = {
        "objective_value": total_obj,
        "s": s_val,
        "scenarios": scenario_solutions
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()