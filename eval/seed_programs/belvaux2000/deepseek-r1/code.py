import gurobipy as gp
from gurobipy import GRB
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
    
    params = data['parameters']
    NI = params['NI']
    NK = params['NK']
    NT = params['NT']
    
    costs = data['costs']
    setup_cost_f = costs['setup_cost_f']
    startup_cost_g = costs['startup_cost_g']
    holding_cost_h = costs['holding_cost_h']
    backlogging_cost_e = costs['backlogging_cost_e']
    
    machine_data = data['machine_data']
    capacity_C = machine_data['capacity_C']
    startup_time_sigma = machine_data['startup_time_sigma']
    
    demand = data['demand']
    
    init_cond = data['initial_conditions']
    initial_stock_s0 = init_cond['initial_stock_s0']
    initial_backlog_r0 = init_cond['initial_backlog_r0']
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    x = {}
    s = {}
    r = {}
    y = {}
    z = {}
    
    for i in range(NI):
        for t in range(NT):
            s_key = (i, t)
            s[s_key] = model.addVar(lb=0, name=f"s_{i}_{t}", vtype=GRB.CONTINUOUS)
            r[s_key] = model.addVar(lb=0, name=f"r_{i}_{t}", vtype=GRB.CONTINUOUS)
            for k in range(NK):
                x_key = (i, k, t)
                x[x_key] = model.addVar(lb=0, name=f"x_{i}_{k}_{t}", vtype=GRB.CONTINUOUS)
                y[x_key] = model.addVar(name=f"y_{i}_{k}_{t}", vtype=GRB.BINARY)
                z[x_key] = model.addVar(name=f"z_{i}_{k}_{t}", vtype=GRB.BINARY)
    
    for i in range(NI):
        for t in range(NT):
            if t == 0:
                prev_inv = initial_stock_s0[i]
                prev_backlog = initial_backlog_r0[i]
            else:
                prev_inv = s[(i, t-1)]
                prev_backlog = r[(i, t-1)]
            total_prod = gp.quicksum(x[(i, k, t)] for k in range(NK))
            model.addConstr(prev_inv - prev_backlog + total_prod == demand[i][t] + s[(i, t)] - r[(i, t)])
    
    for k in range(NK):
        for t in range(NT):
            model.addConstr(gp.quicksum(y[(i, k, t)] for i in range(NI)) <= 1)
    
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                model.addConstr(x[(i, k, t)] <= capacity_C[k] * y[(i, k, t)])
    
    for k in range(NK):
        for t in range(NT):
            total_prod = gp.quicksum(x[(i, k, t)] for i in range(NI))
            total_startup = gp.quicksum(z[(i, k, t)] for i in range(NI))
            model.addConstr(total_prod + startup_time_sigma[k] * total_startup <= capacity_C[k])
    
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                if t == 0:
                    model.addConstr(z[(i, k, t)] >= y[(i, k, t)])
                    model.addConstr(z[(i, k, t)] <= y[(i, k, t)])
                else:
                    model.addConstr(z[(i, k, t)] >= y[(i, k, t)] - y[(i, k, t-1)])
                    model.addConstr(z[(i, k, t)] <= y[(i, k, t)])
                    model.addConstr(z[(i, k, t)] <= 1 - y[(i, k, t-1)])
    
    total_setup_cost = setup_cost_f * gp.quicksum(y[(i, k, t)] for i in range(NI) for k in range(NK) for t in range(NT))
    total_startup_cost = startup_cost_g * gp.quicksum(z[(i, k, t)] for i in range(NI) for k in range(NK) for t in range(NT))
    total_holding_cost = gp.quicksum(holding_cost_h[i] * s[(i, t)] for i in range(NI) for t in range(NT))
    total_backlog_cost = gp.quicksum(backlogging_cost_e[i] * r[(i, t)] for i in range(NI) for t in range(NT))
    total_cost = total_setup_cost + total_startup_cost + total_holding_cost + total_backlog_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        model.optimize(callback)
    else:
        model.optimize()
    
    if model.SolCount > 0:
        production_dict = {}
        stock_dict = {}
        backlog_dict = {}
        setup_dict = {}
        startup_dict = {}
        
        for i in range(NI):
            for t in range(NT):
                stock_key = f"item{i}_period{t}"
                stock_dict[stock_key] = s[(i, t)].X
                backlog_key = f"item{i}_period{t}"
                backlog_dict[backlog_key] = r[(i, t)].X
                for k in range(NK):
                    prod_key = f"item{i}_machine{k}_period{t}"
                    production_dict[prod_key] = x[(i, k, t)].X
                    setup_dict[prod_key] = y[(i, k, t)].X
                    startup_dict[prod_key] = z[(i, k, t)].X
        
        solution = {
            "objective_value": model.ObjVal,
            "production": production_dict,
            "stock": stock_dict,
            "backlog": backlog_dict,
            "setup": setup_dict,
            "startup": startup_dict
        }
    else:
        production_dict = {}
        stock_dict = {}
        backlog_dict = {}
        setup_dict = {}
        startup_dict = {}
        obj_val = 0.0
        
        for i in range(NI):
            s_prev = initial_stock_s0[i]
            r_prev = initial_backlog_r0[i]
            for t in range(NT):
                net_available = s_prev - r_prev
                required = demand[i][t]
                net_after = net_available - required
                if net_after >= 0:
                    s_current = net_after
                    r_current = 0
                else:
                    s_current = 0
                    r_current = -net_after
                
                stock_key = f"item{i}_period{t}"
                stock_dict[stock_key] = s_current
                backlog_key = f"item{i}_period{t}"
                backlog_dict[backlog_key] = r_current
                
                for k in range(NK):
                    key = f"item{i}_machine{k}_period{t}"
                    production_dict[key] = 0.0
                    setup_dict[key] = 0.0
                    startup_dict[key] = 0.0
                
                obj_val += holding_cost_h[i] * s_current
                obj_val += backlogging_cost_e[i] * r_current
                
                s_prev = s_current
                r_prev = r_current
        
        solution = {
            "objective_value": obj_val,
            "production": production_dict,
            "stock": stock_dict,
            "backlog": backlog_dict,
            "setup": setup_dict,
            "startup": startup_dict
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()