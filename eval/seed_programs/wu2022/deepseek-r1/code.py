import argparse
import json
from gurobipy import GRB, Model, quicksum, LinExpr
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
    
    num_items = data['dimensions']['num_items']
    num_periods = data['dimensions']['num_periods']
    num_plants = data['dimensions']['num_plants']
    num_customers = data['dimensions']['num_customers']
    plant_capabilities = data['plant_capabilities']
    demand = data['demand']
    setup_cost = data['setup_cost']
    production_cost = data['production_cost']
    holding_cost = data['holding_cost']
    setup_time = data['setup_time']
    production_time = data['production_time']
    capacity = data['capacity']
    plant_opening_cost = data['plant_opening_cost']
    transportation_cost = data['transportation_cost']
    
    total_demand_i_t = []
    for i in range(num_items):
        item_demands = [0] * num_periods
        for t in range(num_periods):
            for c in range(num_customers):
                item_demands[t] += demand[i][c][t]
        total_demand_i_t.append(item_demands)
    
    T_i_p = [[0] * num_plants for _ in range(num_items)]
    for i in range(num_items):
        for p in plant_capabilities[i]:
            total = 0
            for t in range(num_periods):
                for c in range(num_customers):
                    total += transportation_cost[i][c][p][t] * demand[i][c][t]
            T_i_p[i][p] = total

    total_demand_i = [sum(demands) for demands in total_demand_i_t]

    model = Model("SupplyChain")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', args.time_limit)

    Z_p = [model.addVar(vtype=GRB.BINARY, name=f"Z_{p}") for p in range(num_plants)]
    U_i_p = {}
    Y_i_p_t = {}
    X_i_p_t = {}
    I_i_p_t = {}
    
    for i in range(num_items):
        for p in plant_capabilities[i]:
            U_i_p[i, p] = model.addVar(vtype=GRB.BINARY, name=f"U_{i}_{p}")
            for t in range(num_periods):
                Y_i_p_t[i, p, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{p}_{t}")
                X_i_p_t[i, p, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"X_{i}_{p}_{t}")
                I_i_p_t[i, p, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{p}_{t}")

    for i in range(num_items):
        model.addConstr(quicksum(U_i_p[i, p] for p in plant_capabilities[i]) == 1)

    for i in range(num_items):
        for p in plant_capabilities[i]:
            model.addConstr(U_i_p[i, p] <= Z_p[p])

    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                model.addConstr(X_i_p_t[i, p, t] <= total_demand_i[i] * Y_i_p_t[i, p, t])
                model.addConstr(Y_i_p_t[i, p, t] <= U_i_p[i, p])

    for i in range(num_items):
        for p in plant_capabilities[i]:
            model.addConstr(I_i_p_t[i, p, 0] == X_i_p_t[i, p, 0] - total_demand_i_t[i][0] * U_i_p[i, p])
            for t in range(1, num_periods):
                model.addConstr(I_i_p_t[i, p, t] == I_i_p_t[i, p, t-1] + X_i_p_t[i, p, t] - total_demand_i_t[i][t] * U_i_p[i, p])

    for p in range(num_plants):
        for t in range(num_periods):
            expr = LinExpr()
            for i in range(num_items):
                if p in plant_capabilities[i]:
                    expr.addTerms(setup_time[i][p][t], Y_i_p_t[i, p, t])
                    expr.addTerms(production_time[i][p][t], X_i_p_t[i, p, t])
            model.addConstr(expr <= capacity[p][t] * Z_p[p])

    obj = LinExpr()
    for p in range(num_plants):
        obj.addTerms(plant_opening_cost[p], Z_p[p])
    for i in range(num_items):
        for p in plant_capabilities[i]:
            obj.addTerms(T_i_p[i][p], U_i_p[i, p])
            for t in range(num_periods):
                obj.addTerms(setup_cost[i][p][t], Y_i_p_t[i, p, t])
                obj.addTerms(production_cost[i][p][t], X_i_p_t[i, p, t])
                obj.addTerms(holding_cost[i][p][t], I_i_p_t[i, p, t])
    model.setObjective(obj, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
    
    model.optimize(callback)

    solution = {}
    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal
        solution['Z'] = {str(p): int(Z_p[p].X + 0.5) for p in range(num_plants)}
        solution['U'] = {f"{i}_{p}": int(U_i_p[i, p].X + 0.5) for i in range(num_items) for p in plant_capabilities[i]}
        solution['Y'] = {f"{i}_{p}_{t}": int(Y_i_p_t[i, p, t].X + 0.5) for i in range(num_items) for p in plant_capabilities[i] for t in range(num_periods)}
    else:
        solution['objective_value'] = 0.0
        solution['Z'] = {str(p): 0 for p in range(num_plants)}
        solution['U'] = {f"{i}_{p}": 0 for i in range(num_items) for p in plant_capabilities[i]}
        solution['Y'] = {f"{i}_{p}_{t}": 0 for i in range(num_items) for p in plant_capabilities[i] for t in range(num_periods)}
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()