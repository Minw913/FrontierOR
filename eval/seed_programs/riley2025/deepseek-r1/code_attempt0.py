import json
import argparse
from gurobipy import GRB, Model, quicksum
import os
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--time_limit', type=float, default=1800)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    M = data['num_module_sizes']
    J = data['num_production_sites']
    I = data['num_demand_sites']
    T = data['num_time_periods']
    S = data['num_scenarios']
    K = data['units_per_module_size']
    module_sizes = data['module_sizes']
    module_units = data['module_units']
    production_costs = data['production_costs']
    relocation_costs = data['relocation_costs']
    penalty_costs = data['penalty_costs']
    scenarios = data['scenarios']
    scenario_probabilities = data['scenario_probabilities']
    u_max = data['u_max']

    cap_cost = [[0]*K for _ in range(M)]
    for unit in module_units:
        m = unit['module_size_id']
        k = unit['unit_id']
        cap_cost[m][k] = unit['capital_cost']

    model = Model("Stochastic_Manufacturing")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    y = {}
    for m in range(M):
        for k in range(K):
            y[m,k] = model.addVar(vtype=GRB.BINARY, name=f"y_{m}_{k}")

    z = {}
    x = {}
    q = {}
    v_depot = {}
    v = {}
    w = {}
    w_fd = {}
    w_td = {}

    for s in range(S):
        for m in range(M):
            z[m,s] = model.addVar(vtype=GRB.INTEGER, name=f"z_{m}_{s}")
        for i in range(I):
            for t in range(T):
                q[i,t,s] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"q_{i}_{t}_{s}")
                for j in range(J):
                    x[i,j,t,s] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}_{t}_{s}")
        for m in range(M):
            for t in range(T):
                v_depot[m,t,s] = model.addVar(vtype=GRB.INTEGER, name=f"v_depot_{m}_{t}_{s}")
                for j in range(J):
                    v[j,m,t,s] = model.addVar(vtype=GRB.INTEGER, name=f"v_{j}_{m}_{t}_{s}")
                    w_fd[j,m,t,s] = model.addVar(vtype=GRB.INTEGER, name=f"w_fd_{j}_{m}_{t}_{s}")
                    w_td[j,m,t,s] = model.addVar(vtype=GRB.INTEGER, name=f"w_td_{j}_{m}_{t}_{s}")
        for j in range(J):
            for jp in range(J):
                if j != jp:
                    for m in range(M):
                        for t in range(T):
                            w[j,jp,m,t,s] = model.addVar(vtype=GRB.INTEGER, name=f"w_{j}_{jp}_{m}_{t}_{s}")

    for s in range(S):
        for m in range(M):
            Y_m = quicksum(y[m,k] for k in range(K))
            model.addConstr(z[m,s] == Y_m, name=f"z_def_{m}_{s}")
        for m in range(M):
            model.addConstr(v_depot[m,0,s] == z[m,s] - quicksum(w_fd[j,m,0,s] for j in range(J)) + quicksum(w_td[j,m,0,s] for j in range(J)), name=f"depot_balance_0_{m}_{s}")
        for j in range(J):
            for m in range(M):
                expr = w_fd[j,m,0,s] - w_td[j,m,0,s]
                for jp in range(J):
                    if jp != j:
                        expr += w[jp,j,m,0,s] - w[j,jp,m,0,s]
                model.addConstr(v[j,m,0,s] == expr, name=f"site_balance_{j}_{m}_0_{s}")
        for t in range(1, T):
            for m in range(M):
                model.addConstr(v_depot[m,t,s] == v_depot[m,t-1,s] - quicksum(w_fd[j,m,t,s] for j in range(J)) + quicksum(w_td[j,m,t,s] for j in range(J)), name=f"depot_balance_{t}_{m}_{s}")
            for j in range(J):
                for m in range(M):
                    expr = v[j,m,t-1,s] + w_fd[j,m,t,s] - w_td[j,m,t,s]
                    for jp in range(J):
                        if jp != j:
                            expr += w[jp,j,m,t,s] - w[j,jp,m,t,s]
                    model.addConstr(v[j,m,t,s] == expr, name=f"site_balance_{j}_{m}_{t}_{s}")
        for i in range(I):
            for t in range(T):
                model.addConstr(quicksum(x[i,j,t,s] for j in range(J)) + q[i,t,s] == 1, name=f"demand_{i}_{t}_{s}")
        for j in range(J):
            for t in range(T):
                total_production = quicksum(scenarios[s]['demands'][i][t] * x[i,j,t,s] for i in range(I))
                model.addConstr(total_production <= u_max, name=f"u_max_{j}_{t}_{s}")
                total_capacity = quicksum(module_sizes[m]['capacity'] * v[j,m,t,s] for m in range(M))
                model.addConstr(total_production <= total_capacity, name=f"capacity_{j}_{t}_{s}")

    capital_cost = quicksum(cap_cost[m][k] * y[m,k] for m in range(M) for k in range(K))
    expected_operating_cost = 0
    for s in range(S):
        prob_s = scenario_probabilities[s]
        prod_trans_cost = quicksum(
            production_costs[i][j] * scenarios[s]['demands'][i][t] * x[i,j,t,s]
            for i in range(I) for j in range(J) for t in range(T)
        )
        reloc_cost = quicksum(
            relocation_costs[j][jp][m] * w[j,jp,m,t,s]
            for j in range(J) for jp in range(J) if j != jp
            for m in range(M) for t in range(T)
        )
        penalty_cost = quicksum(
            penalty_costs[i][t] * scenarios[s]['demands'][i][t] * q[i,t,s]
            for i in range(I) for t in range(T)
        )
        scenario_cost = prod_trans_cost + reloc_cost + penalty_cost
        expected_operating_cost += prob_s * scenario_cost
    total_cost = capital_cost + expected_operating_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    if logger:
        model._last_obj = None
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if model._last_obj is None or objval < model._last_obj:
                    model._last_obj = objval
                    logger.log(objval)
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        objective_value = model.objVal
        y_sol = {}
        for m in range(M):
            for k in range(K):
                key = f"y_{m}_{k}"
                y_sol[key] = y[m,k].X

        model_vars = {}
        for s in range(S):
            for m in range(M):
                key = f"z[{m},{s}]"
                model_vars[key] = z[m,s].X
            for i in range(I):
                for t in range(T):
                    key_q = f"q[{i},{t},{s}]"
                    model_vars[key_q] = q[i,t,s].X
                    for j in range(J):
                        key_x = f"x[{i},{j},{t},{s}]"
                        model_vars[key_x] = x[i,j,t,s].X
            for m in range(M):
                for t in range(T):
                    key_vd = f"v_depot[{m},{t},{s}]"
                    model_vars[key_vd] = v_depot[m,t,s].X
            for j in range(J):
                for m in range(M):
                    for t in range(T):
                        key_v = f"v[{j},{m},{t},{s}]"
                        model_vars[key_v] = v[j,m,t,s].X
                        key_wfd = f"w_fd[{j},{m},{t},{s}]"
                        model_vars[key_wfd] = w_fd[j,m,t,s].X
                        key_wtd = f"w_td[{j},{m},{t},{s}]"
                        model_vars[key_wtd] = w_td[j,m,t,s].X
            for j in range(J):
                for jp in range(J):
                    if j != jp:
                        for m in range(M):
                            for t in range(T):
                                key_w = f"w[{j},{jp},{m},{t},{s}]"
                                model_vars[key_w] = w[j,jp,m,t,s].X
    else:
        objective_value = 0
        for m in range(M):
            for k in range(K):
                objective_value += cap_cost[m][k] * 0
        for s in range(S):
            prob_s = scenario_probabilities[s]
            penalty_cost_val = 0
            for i in range(I):
                for t in range(T):
                    penalty_cost_val += penalty_costs[i][t] * scenarios[s]['demands'][i][t] * 1
            objective_value += prob_s * penalty_cost_val
        y_sol = {f"y_{m}_{k}": 0 for m in range(M) for k in range(K)}
        model_vars = {}
        for s in range(S):
            for m in range(M):
                key = f"z[{m},{s}]"
                model_vars[key] = 0.0
            for i in range(I):
                for t in range(T):
                    key_q = f"q[{i},{t},{s}]"
                    model_vars[key_q] = 1.0
                    for j in range(J):
                        key_x = f"x[{i},{j},{t},{s}]"
                        model_vars[key_x] = 0.0

    solution_dict = {
        "objective_value": objective_value,
        "model_variables": model_vars,
        "y": y_sol,
        "model_variable_name_patterns": {
            "y[m,k]": "m=0..num_module_sizes-1, k=0..units_per_module_size-1",
            "z[m,s]": "module size m, scenario s",
            "x[i,j,t,s]": "demand site i, production site j, period t, scenario s",
            "q[i,t,s]": "unmet-demand fraction",
            "v[j,m,t,s]": "units at production site",
            "v_depot[m,t,s]": "units at depot",
            "w[j,jp,m,t,s]": "relocation between distinct production sites j and jp",
            "w_fd[j,m,t,s]": "depot-to-production-site relocation",
            "w_td[j,m,t,s]": "production-site-to-depot relocation"
        }
    }

    if logger:
        logger.log_solution(objective_value, solution_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == "__main__":
    main()