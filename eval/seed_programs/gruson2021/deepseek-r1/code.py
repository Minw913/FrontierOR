import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    T = instance['num_periods']
    W = instance['num_warehouses']
    R = instance['num_retailers']
    S_val = instance['num_scenarios']
    retailer_to_warehouse = instance['retailer_to_warehouse']
    scenario_probabilities = instance['scenario_probabilities']
    holding_cost_plant = instance['holding_cost_plant']
    holding_cost_warehouse = instance['holding_cost_warehouse']
    holding_costs_retailers = instance['holding_costs_retailers']
    setup_costs_plant = instance['setup_costs_plant']
    setup_costs_warehouses = instance['setup_costs_warehouses']
    setup_costs_retailers = instance['setup_costs_retailers']
    demands = instance['demands']

    cum_demands = []
    for s in range(S_val):
        scen_demands = demands[s]
        cum_scen = []
        for k in range(R):
            cum_k = [0] * T
            if T > 0:
                cum_k[T-1] = scen_demands[k][T-1]
                for t in range(T-2, -1, -1):
                    cum_k[t] = scen_demands[k][t] + cum_k[t+1]
            cum_scen.append(cum_k)
        cum_demands.append(cum_scen)

    total_demand_plant_t = []
    for s in range(S_val):
        per_t = [0] * T
        for t in range(T):
            total = 0
            for k in range(R):
                total += cum_demands[s][k][t]
            per_t[t] = total
        total_demand_plant_t.append(per_t)

    total_demand_warehouse_j_t = []
    for s in range(S_val):
        per_warehouse = []
        for j in range(W):
            per_t = [0] * T
            for t in range(T):
                total = 0
                for k in range(R):
                    if retailer_to_warehouse[k] == j:
                        total += cum_demands[s][k][t]
                per_t[t] = total
            per_warehouse.append(per_t)
        total_demand_warehouse_j_t.append(per_warehouse)

    total_demand_retailer_k_t = cum_demands

    model = gp.Model("Supply_Chain_Optimization")
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)

    y0 = model.addVars(T, vtype=GRB.BINARY, name="y0")
    yw = {}
    for j in range(W):
        yw[j] = model.addVars(T, vtype=GRB.BINARY, name=f"yw_{j}")
    yr = {}
    for k in range(R):
        yr[k] = model.addVars(T, vtype=GRB.BINARY, name=f"yr_{k}")

    model.addConstr(y0[0] == 1, name="initial_plant_setup")
    for j in range(W):
        model.addConstr(yw[j][0] == 1, name=f"initial_warehouse_{j}_setup")
    for k in range(R):
        model.addConstr(yr[k][0] == 1, name=f"initial_retailer_{k}_setup")

    X0 = {}
    S0 = {}
    I0 = {}
    Xw = {}
    Sw = {}
    Iw = {}
    Xr = {}
    Ir = {}

    for s in range(S_val):
        X0_s = model.addVars(T, lb=0, name=f"X0_s{s}")
        S0_s = model.addVars(T, lb=0, name=f"S0_s{s}")
        I0_s = model.addVars(T, lb=0, name=f"I0_s{s}")
        X0[s] = X0_s
        S0[s] = S0_s
        I0[s] = I0_s

        Xw_s = {}
        Sw_s = {}
        Iw_s = {}
        for j in range(W):
            Xw_s[j] = model.addVars(T, lb=0, name=f"Xw_s{s}_j{j}")
            Sw_s[j] = model.addVars(T, lb=0, name=f"Sw_s{s}_j{j}")
            Iw_s[j] = model.addVars(T, lb=0, name=f"Iw_s{s}_j{j}")
        Xw[s] = Xw_s
        Sw[s] = Sw_s
        Iw[s] = Iw_s

        Xr_s = {}
        Ir_s = {}
        for k in range(R):
            Xr_s[k] = model.addVars(T, lb=0, name=f"Xr_s{s}_k{k}")
            Ir_s[k] = model.addVars(T, lb=0, name=f"Ir_s{s}_k{k}")
        Xr[s] = Xr_s
        Ir[s] = Ir_s

        for t in range(T):
            if t == 0:
                model.addConstr(X0_s[t] == S0_s[t] + I0_s[t], name=f"plant_bal_s{s}_t{t}")
            else:
                model.addConstr(I0_s[t-1] + X0_s[t] == S0_s[t] + I0_s[t], name=f"plant_bal_s{s}_t{t}")

        for j in range(W):
            for t in range(T):
                if t == 0:
                    model.addConstr(Xw_s[j][t] == Sw_s[j][t] + Iw_s[j][t], name=f"wh_bal_s{s}_j{j}_t{t}")
                else:
                    model.addConstr(Iw_s[j][t-1] + Xw_s[j][t] == Sw_s[j][t] + Iw_s[j][t], name=f"wh_bal_s{s}_j{j}_t{t}")

        for k in range(R):
            for t in range(T):
                if t == 0:
                    model.addConstr(Xr_s[k][t] == demands[s][k][t] + Ir_s[k][t], name=f"ret_bal_s{s}_k{k}_t{t}")
                else:
                    model.addConstr(Ir_s[k][t-1] + Xr_s[k][t] == demands[s][k][t] + Ir_s[k][t], name=f"ret_bal_s{s}_k{k}_t{t}")

        for t in range(T):
            model.addConstr(S0_s[t] == gp.quicksum(Xw_s[j][t] for j in range(W)), name=f"plant_ship_s{s}_t{t}")

        for j in range(W):
            retailers_j = [k for k in range(R) if retailer_to_warehouse[k] == j]
            for t in range(T):
                model.addConstr(Sw_s[j][t] == gp.quicksum(Xr_s[k][t] for k in retailers_j), name=f"wh_ship_s{s}_j{j}_t{t}")

        for t in range(T):
            model.addConstr(X0_s[t] <= total_demand_plant_t[s][t] * y0[t], name=f"setup_plant_s{s}_t{t}")

        for j in range(W):
            for t in range(T):
                model.addConstr(Xw_s[j][t] <= total_demand_warehouse_j_t[s][j][t] * yw[j][t], name=f"setup_wh_s{s}_j{j}_t{t}")

        for k in range(R):
            for t in range(T):
                model.addConstr(Xr_s[k][t] <= total_demand_retailer_k_t[s][k][t] * yr[k][t], name=f"setup_ret_s{s}_k{k}_t{t}")

    setup_cost = gp.quicksum(setup_costs_plant[t] * y0[t] for t in range(T))
    for j in range(W):
        setup_cost += gp.quicksum(setup_costs_warehouses[j][t] * yw[j][t] for t in range(T))
    for k in range(R):
        setup_cost += gp.quicksum(setup_costs_retailers[k][t] * yr[k][t] for t in range(T))

    holding_cost = 0
    for s in range(S_val):
        prob = scenario_probabilities[s]
        plant_holding = holding_cost_plant * gp.quicksum(I0[s][t] for t in range(T))
        warehouse_holding = holding_cost_warehouse * gp.quicksum(Iw[s][j][t] for j in range(W) for t in range(T))
        retailer_holding = gp.quicksum(holding_costs_retailers[k] * gp.quicksum(Ir[s][k][t] for t in range(T)) for k in range(R))
        holding_cost += prob * (plant_holding + warehouse_holding + retailer_holding)

    total_cost = setup_cost + holding_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(objval)

    model.setParam('TimeLimit', args.time_limit)
    model.optimize(callback if logger else None)

    if model.SolCount > 0:
        setup_vars_dict = {}
        for t in range(T):
            setup_vars_dict[f'y_0_{t}'] = float(y0[t].X)

        for j in range(W):
            for t in range(T):
                setup_vars_dict[f'y_{j+1}_{t}'] = float(yw[j][t].X)

        for k in range(R):
            for t in range(T):
                setup_vars_dict[f'y_{W+1+k}_{t}'] = float(yr[k][t].X)

        production_plant_dict = {}
        for s in range(S_val):
            for t in range(T):
                key = f"{t}_{s}"
                production_plant_dict[key] = float(X0[s][t].X)

        delivery_warehouse_dict = {}
        for s in range(S_val):
            for j in range(W):
                for t in range(T):
                    key = f"{j}_{t}_{s}"
                    delivery_warehouse_dict[key] = float(Xw[s][j][t].X)

        delivery_retailer_dict = {}
        for s in range(S_val):
            for k in range(R):
                for t in range(T):
                    key = f"{k}_{t}_{s}"
                    delivery_retailer_dict[key] = float(Xr[s][k][t].X)

        inventory_plant_dict = {}
        for s in range(S_val):
            for t in range(T):
                key = f"{t}_{s}"
                inventory_plant_dict[key] = float(I0[s][t].X)

        inventory_warehouse_dict = {}
        for s in range(S_val):
            for j in range(W):
                for t in range(T):
                    key = f"{j}_{t}_{s}"
                    inventory_warehouse_dict[key] = float(Iw[s][j][t].X)

        inventory_retailer_dict = {}
        for s in range(S_val):
            for k in range(R):
                for t in range(T):
                    key = f"{k}_{t}_{s}"
                    inventory_retailer_dict[key] = float(Ir[s][k][t].X)

        solution = {
            "objective_value": model.ObjVal,
            "setup_variables": setup_vars_dict,
            "production_plant": production_plant_dict,
            "delivery_warehouse": delivery_warehouse_dict,
            "delivery_retailer": delivery_retailer_dict,
            "inventory_plant": inventory_plant_dict,
            "inventory_warehouse": inventory_warehouse_dict,
            "inventory_retailer": inventory_retailer_dict
        }
    else:
        setup_vars_dict = {}
        for t in range(T):
            setup_vars_dict[f'y_0_{t}'] = 1.0 if t == 0 else 0.0

        for j in range(W):
            for t in range(T):
                setup_vars_dict[f'y_{j+1}_{t}'] = 1.0 if t == 0 else 0.0

        for k in range(R):
            for t in range(T):
                setup_vars_dict[f'y_{W+1+k}_{t}'] = 1.0 if t == 0 else 0.0

        production_plant_dict = {}
        delivery_warehouse_dict = {}
        delivery_retailer_dict = {}
        inventory_plant_dict = {}
        inventory_warehouse_dict = {}
        inventory_retailer_dict = {}

        for s in range(S_val):
            for t in range(T):
                key = f"{t}_{s}"
                production_plant_dict[key] = 0.0
                inventory_plant_dict[key] = 0.0

        for s in range(S_val):
            for j in range(W):
                for t in range(T):
                    key = f"{j}_{t}_{s}"
                    delivery_warehouse_dict[key] = 0.0
                    inventory_warehouse_dict[key] = 0.0

        for s in range(S_val):
            for k in range(R):
                for t in range(T):
                    key = f"{k}_{t}_{s}"
                    delivery_retailer_dict[key] = 0.0
                    inventory_retailer_dict[key] = 0.0

        solution = {
            "objective_value": float('inf'),
            "setup_variables": setup_vars_dict,
            "production_plant": production_plant_dict,
            "delivery_warehouse": delivery_warehouse_dict,
            "delivery_retailer": delivery_retailer_dict,
            "inventory_plant": inventory_plant_dict,
            "inventory_warehouse": inventory_warehouse_dict,
            "inventory_retailer": inventory_retailer_dict
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()