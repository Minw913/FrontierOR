import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    n = inst['n']
    T = inst['T']
    m = inst['m']
    Q = inst['Q']
    C = inst['C']
    f_cost = inst['f']
    u_cost = inst['u']
    trans_costs = inst['transportation_costs']
    h = inst['h']
    L = inst['L']
    I0 = inst['I0']
    sigma = inst['sigma']
    n_scen = inst['n_scenarios']
    scen_probs = inst['scenario_probabilities']
    demand_scenarios = inst['demand_scenarios']
    d_bar = inst.get('d_bar', None)

    if n_scen == 0 or not demand_scenarios:
        if d_bar is not None:
            n_scen = 1
            scen_probs = [1.0]
            demand_scenarios = [d_bar]
        else:
            n_scen = 1
            scen_probs = [1.0]
            demand_scenarios = [[[0] * T for _ in range(n)]]

    total_prob = sum(scen_probs) if scen_probs else 1.0
    if total_prob > 0:
        scen_probs = [p / total_prob for p in scen_probs]
    else:
        scen_probs = [1.0 / n_scen] * n_scen if n_scen > 0 else [1.0]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    customers = list(range(1, n + 1))
    nodes = [0] + customers
    periods = list(range(1, T + 1))
    vehicles = list(range(1, m + 1)) if m > 0 else []
    scenarios = list(range(n_scen))

    model = gp.Model('PRP')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # === Variables ===
    y = model.addVars(periods, vtype=GRB.BINARY, name='y')

    z, x, pos, q = {}, {}, {}, {}
    if vehicles:
        z_keys = [(i, k, t) for i in nodes for k in vehicles for t in periods]
        z = model.addVars(z_keys, vtype=GRB.BINARY, name='z')

        x_keys = [(i, j, k, t) for i in nodes for j in nodes if i != j
                  for k in vehicles for t in periods]
        x = model.addVars(x_keys, vtype=GRB.INTEGER, lb=0, ub=2, name='x')

        if customers:
            pos_keys = [(i, k, t) for i in customers for k in vehicles for t in periods]
            pos = model.addVars(pos_keys, vtype=GRB.CONTINUOUS, lb=0, ub=n, name='pos')

            q_keys = [(i, k, t, omega) for i in customers for k in vehicles
                      for t in periods for omega in scenarios]
            q = model.addVars(q_keys, vtype=GRB.CONTINUOUS, lb=0, name='q')

    p = model.addVars(periods, scenarios, vtype=GRB.CONTINUOUS, lb=0, name='p')
    I_vars = model.addVars(nodes, periods, scenarios, vtype=GRB.CONTINUOUS, lb=0, name='I')
    e = model.addVars(customers, periods, scenarios, vtype=GRB.CONTINUOUS, lb=0, name='e')

    model.update()

    # === Routing Constraints ===
    if vehicles and customers:
        for t in periods:
            model.addConstrs(
                gp.quicksum(z[i, k, t] for k in vehicles) <= 1 for i in customers)

            for k in vehicles:
                model.addConstrs(z[i, k, t] <= z[0, k, t] for i in customers)

                for i in customers:
                    model.addConstr(
                        gp.quicksum(x[j, i, k, t] for j in nodes if j != i) == z[i, k, t])
                    model.addConstr(
                        gp.quicksum(x[i, j, k, t] for j in nodes if j != i) == z[i, k, t])

                model.addConstr(
                    gp.quicksum(x[0, j, k, t] for j in customers) == z[0, k, t])
                model.addConstr(
                    gp.quicksum(x[j, 0, k, t] for j in customers) == z[0, k, t])

                model.addConstrs(pos[i, k, t] >= z[i, k, t] for i in customers)
                model.addConstrs(pos[i, k, t] <= n * z[i, k, t] for i in customers)

                for i in customers:
                    for j in customers:
                        if i < j:
                            model.addConstr(
                                pos[i, k, t] - pos[j, k, t] + n * x[i, j, k, t] <=
                                n - 1 + n * (2 - z[i, k, t] - z[j, k, t]))
                            model.addConstr(
                                pos[j, k, t] - pos[i, k, t] + n * x[j, i, k, t] <=
                                n - 1 + n * (2 - z[j, k, t] - z[i, k, t]))

        for t in periods:
            for k in range(1, m):
                model.addConstr(z[0, k, t] >= z[0, k + 1, t])

    # === Production Capacity ===
    model.addConstrs(p[t, omega] <= C * y[t] for t in periods for omega in scenarios)

    # === Plant Inventory Balance ===
    for t in periods:
        for omega in scenarios:
            prev_inv = I0[0] if t == 1 else I_vars[0, t - 1, omega]
            td = (gp.quicksum(q[i, k, t, omega] for i in customers for k in vehicles)
                  if vehicles and customers else 0)
            model.addConstr(prev_inv + p[t, omega] == td + I_vars[0, t, omega])
    model.addConstrs(I_vars[0, t, omega] <= L[0]
                     for t in periods for omega in scenarios)

    # === Customer Inventory Balance ===
    for i in customers:
        for t in periods:
            for omega in scenarios:
                d = demand_scenarios[omega][i - 1][t - 1]
                prev_inv = I0[i] if t == 1 else I_vars[i, t - 1, omega]
                deliv = (gp.quicksum(q[i, k, t, omega] for k in vehicles)
                         if vehicles else 0)
                model.addConstr(
                    prev_inv + deliv + e[i, t, omega] == d + I_vars[i, t, omega])
                model.addConstr(I_vars[i, t, omega] + d <= L[i])

    # === Vehicle Capacity and Delivery-Visit Linking ===
    if vehicles and customers:
        for k in vehicles:
            for t in periods:
                for omega in scenarios:
                    model.addConstr(
                        gp.quicksum(q[i, k, t, omega] for i in customers) <= Q * z[0, k, t])

        model.addConstrs(
            q[i, k, t, omega] <= Q * z[i, k, t]
            for i in customers for k in vehicles for t in periods for omega in scenarios)

        for i in customers:
            for t in periods:
                for omega in scenarios:
                    prev_inv = I0[i] if t == 1 else I_vars[i, t - 1, omega]
                    model.addConstr(
                        prev_inv + gp.quicksum(q[i, k, t, omega] for k in vehicles) <= L[i])

        for i in customers:
            for t in periods:
                for omega in scenarios:
                    fd = sum(demand_scenarios[omega][i - 1][tp - 1]
                             for tp in range(t, T + 1))
                    if fd > 0:
                        model.addConstrs(
                            q[i, k, t, omega] <= fd for k in vehicles)

    # === Objective ===
    routing_cost = gp.LinExpr(0)
    if vehicles and customers:
        routing_cost = gp.quicksum(
            trans_costs[i][j] * x[i, j, k, t]
            for i in nodes for j in nodes if i != j
            for k in vehicles for t in periods)
    setup_cost = gp.quicksum(f_cost * y[t] for t in periods)
    scenario_cost = gp.quicksum(
        scen_probs[omega] * (
            gp.quicksum(u_cost * p[t, omega] for t in periods) +
            gp.quicksum(h[i] * I_vars[i, t, omega] for i in nodes for t in periods) +
            gp.quicksum(sigma[i - 1] * e[i, t, omega]
                        for i in customers for t in periods))
        for omega in scenarios)
    model.setObjective(setup_cost + routing_cost + scenario_cost, GRB.MINIMIZE)

    # === Callback ===
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # === Solve ===
    elapsed = time.time() - start_time
    model.Params.TimeLimit = max(1.0, float(args.time_limit) - elapsed - 3.0)
    model.optimize(callback)

    # === Extract Solution ===
    model_vars = {}
    y_dict, z_dict, x_dict = {}, {}, {}

    if model.SolCount > 0:
        obj_val = model.ObjVal
        for t in periods:
            if round(y[t].X) > 0.5:
                y_dict[str(t)] = 1
                model_vars[f'y_{t}'] = 1.0
        if vehicles and customers:
            for i in nodes:
                for k in vehicles:
                    for t in periods:
                        if round(z[i, k, t].X) > 0.5:
                            z_dict[f'{i}_{k}_{t}'] = 1
                            model_vars[f'z_{i}_{k}_{t}'] = 1.0
            for i in nodes:
                for j in nodes:
                    if i != j:
                        for k in vehicles:
                            for t in periods:
                                v = x[i, j, k, t].X
                                if v > 0.5:
                                    iv = int(round(v))
                                    x_dict[f'{i}_{j}_{k}_{t}'] = iv
                                    model_vars[f'x_{i}_{j}_{k}_{t}'] = float(iv)
        for t in periods:
            for omega in scenarios:
                v = p[t, omega].X
                if v > 1e-6:
                    model_vars[f'p_{t}_{omega}'] = v
        for i in nodes:
            for t in periods:
                for omega in scenarios:
                    v = I_vars[i, t, omega].X
                    if v > 1e-6:
                        model_vars[f'I_{i}_{t}_{omega}'] = v
        for i in customers:
            for t in periods:
                for omega in scenarios:
                    v = e[i, t, omega].X
                    if v > 1e-6:
                        model_vars[f'e_{i}_{t}_{omega}'] = v
        if vehicles and customers:
            for i in customers:
                for k in vehicles:
                    for t in periods:
                        for omega in scenarios:
                            v = q[i, k, t, omega].X
                            if v > 1e-6:
                                model_vars[f'q_{i}_{k}_{t}_{omega}'] = v
        if logger:
            logger.log_solution(obj_val, {
                'objective_value': obj_val,
                'model_variables': model_vars,
                'y': y_dict, 'z': z_dict, 'x': x_dict})
    else:
        # Trivial fallback: no production, no delivery, all demand unmet
        obj_val = 0.0
        for omega in scenarios:
            sc = 0.0
            for t in periods:
                model_vars[f'I_0_{t}_{omega}'] = float(I0[0])
                sc += h[0] * I0[0]
            ci = {i: I0[i] for i in customers}
            for t in periods:
                for i in customers:
                    d = demand_scenarios[omega][i - 1][t - 1]
                    if ci[i] >= d:
                        ci[i] -= d
                        ev = 0.0
                    else:
                        ev = float(d - ci[i])
                        ci[i] = 0
                    if ci[i] > 1e-6:
                        model_vars[f'I_{i}_{t}_{omega}'] = float(ci[i])
                        sc += h[i] * ci[i]
                    if ev > 1e-6:
                        model_vars[f'e_{i}_{t}_{omega}'] = ev
                        sc += sigma[i - 1] * ev
            obj_val += scen_probs[omega] * sc

    solution = {
        'objective_value': float(obj_val),
        'model_variables': model_vars,
        'y': y_dict,
        'z': z_dict,
        'x': x_dict,
        'model_variable_name_patterns': {
            'y_{t}': 'setup in one-based period t',
            'z_{i}_{k}_{t}': 'visit node i by vehicle k in period t',
            'x_{i}_{j}_{k}_{t}': 'edge traversal',
            'p_{t}_{omega}': 'scenario production',
            'I_{i}_{t}_{omega}': 'scenario inventory',
            'e_{i}_{t}_{omega}': 'scenario emergency quantity',
            'q_{i}_{k}_{t}_{omega}': 'scenario delivery quantity'
        }
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()