import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def build_fallback_solution(inst):
    T = inst['num_periods']; W = inst['num_warehouses']; R = inst['num_retailers']; S = inst['num_scenarios']
    r2w = inst['retailer_to_warehouse']; probs = inst['scenario_probabilities']
    hp = inst['holding_cost_plant']; hw = inst['holding_cost_warehouse']; hr = inst['holding_costs_retailers']
    cp = inst['setup_costs_plant']; cw = inst['setup_costs_warehouses']; cr = inst['setup_costs_retailers']
    dem = inst['demands']; init_inv = inst.get('initial_inventory', 0)

    setup_vars = {}
    for t in range(T): setup_vars[f'y_0_{t}'] = 1
    for w in range(W):
        for t in range(T): setup_vars[f'y_{1+w}_{t}'] = 1
    for r in range(R):
        for t in range(T): setup_vars[f'y_{1+W+r}_{t}'] = 1

    sol = {'setup_variables': setup_vars,
           'production_plant': {}, 'delivery_warehouse': {}, 'delivery_retailer': {},
           'inventory_plant': {}, 'inventory_warehouse': {}, 'inventory_retailer': {}}

    obj = sum(cp[t] for t in range(T))
    obj += sum(cw[w][t] for w in range(W) for t in range(T))
    obj += sum(cr[r][t] for r in range(R) for t in range(T))
    holding = 0.0
    for s in range(S):
        p = probs[s]
        for t in range(T):
            holding += p * hp * init_inv
            for w in range(W): holding += p * hw * init_inv
            for r in range(R): holding += p * hr[r] * init_inv
    obj += holding

    for s in range(S):
        for t in range(T):
            td = sum(dem[s][r][t] for r in range(R))
            sol['production_plant'][f'{t}_{s}'] = float(td)
            sol['inventory_plant'][f'{t}_{s}'] = float(init_inv)
            for w in range(W):
                wd = sum(dem[s][r][t] for r in range(R) if r2w[r] == w)
                sol['delivery_warehouse'][f'{w}_{t}_{s}'] = float(wd)
                sol['inventory_warehouse'][f'{w}_{t}_{s}'] = float(init_inv)
            for r in range(R):
                sol['delivery_retailer'][f'{r}_{t}_{s}'] = float(dem[s][r][t])
                sol['inventory_retailer'][f'{r}_{t}_{s}'] = float(init_inv)
    sol['objective_value'] = float(obj)
    return obj, sol


def compute_objective(sol, inst):
    T = inst['num_periods']; W = inst['num_warehouses']; R = inst['num_retailers']; S = inst['num_scenarios']
    probs = inst['scenario_probabilities']
    hp = inst['holding_cost_plant']; hw = inst['holding_cost_warehouse']; hr = inst['holding_costs_retailers']
    cp = inst['setup_costs_plant']; cw = inst['setup_costs_warehouses']; cr = inst['setup_costs_retailers']
    setup = sol['setup_variables']
    obj = 0.0
    for t in range(T): obj += cp[t] * setup[f'y_0_{t}']
    for w in range(W):
        for t in range(T): obj += cw[w][t] * setup[f'y_{1+w}_{t}']
    for r in range(R):
        for t in range(T): obj += cr[r][t] * setup[f'y_{1+W+r}_{t}']
    for s in range(S):
        p = probs[s]
        for t in range(T):
            obj += p * hp * sol['inventory_plant'][f'{t}_{s}']
            for w in range(W):
                obj += p * hw * sol['inventory_warehouse'][f'{w}_{t}_{s}']
            for r in range(R):
                obj += p * hr[r] * sol['inventory_retailer'][f'{r}_{t}_{s}']
    return obj


def make_sol_from_aggregate(yp_v, yw_v, yr_v, P_v, SW_v, SR_v, IP_v, IW_v, IR_v, T, W, R, S):
    setup_vars = {}
    for t in range(T): setup_vars[f'y_0_{t}'] = int(round(yp_v[t]))
    for w in range(W):
        for t in range(T): setup_vars[f'y_{1+w}_{t}'] = int(round(yw_v[w, t]))
    for r in range(R):
        for t in range(T): setup_vars[f'y_{1+W+r}_{t}'] = int(round(yr_v[r, t]))
    return {
        'setup_variables': setup_vars,
        'production_plant': {f'{t}_{s}': float(P_v[t, s]) for t in range(T) for s in range(S)},
        'delivery_warehouse': {f'{w}_{t}_{s}': float(SW_v[w, t, s]) for w in range(W) for t in range(T) for s in range(S)},
        'delivery_retailer': {f'{r}_{t}_{s}': float(SR_v[r, t, s]) for r in range(R) for t in range(T) for s in range(S)},
        'inventory_plant': {f'{t}_{s}': float(IP_v[t, s]) for t in range(T) for s in range(S)},
        'inventory_warehouse': {f'{w}_{t}_{s}': float(IW_v[w, t, s]) for w in range(W) for t in range(T) for s in range(S)},
        'inventory_retailer': {f'{r}_{t}_{s}': float(IR_v[r, t, s]) for r in range(R) for t in range(T) for s in range(S)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()
    with open(args.instance_path) as f:
        inst = json.load(f)

    T = inst['num_periods']; W = inst['num_warehouses']; R = inst['num_retailers']; S = inst['num_scenarios']
    r2w = inst['retailer_to_warehouse']; probs = inst['scenario_probabilities']
    hp = inst['holding_cost_plant']; hw = inst['holding_cost_warehouse']; hr = inst['holding_costs_retailers']
    cp = inst['setup_costs_plant']; cw = inst['setup_costs_warehouses']; cr = inst['setup_costs_retailers']
    dem = inst['demands']; init_inv = inst.get('initial_inventory', 0)
    initial_setups = inst.get('initial_setups_imposed', True)

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    fb_obj, fb_sol = build_fallback_solution(inst)
    if logger: logger.log_solution(fb_obj, fb_sol)
    best_obj = fb_obj
    best_sol = fb_sol

    extended_size = (T * (T + 1) // 2) * R * S
    use_extended = (init_inv == 0) and (extended_size < 60000)

    elapsed = time.time() - start_time
    remaining = max(2.0, float(args.time_limit) - elapsed - 3.0)

    try:
        model = gp.Model()
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining
        model.Params.OutputFlag = 0

        y_p = model.addVars(T, vtype=GRB.BINARY, name='yp')
        y_w = model.addVars(W, T, vtype=GRB.BINARY, name='yw')
        y_r = model.addVars(R, T, vtype=GRB.BINARY, name='yr')

        if initial_setups:
            model.addConstr(y_p[0] == 1)
            for w in range(W): model.addConstr(y_w[w, 0] == 1)
            for r in range(R): model.addConstr(y_r[r, 0] == 1)

        best_holder = [best_obj, best_sol]

        if use_extended:
            idx_z = [(t, r, d, s) for s in range(S) for r in range(R) for d in range(T) for t in range(d + 1)]
            zp = model.addVars(idx_z, lb=0, ub=1, name='zp')
            zw = model.addVars(idx_z, lb=0, ub=1, name='zw')
            zr = model.addVars(idx_z, lb=0, ub=1, name='zr')

            # Setup linking (only for items with positive demand)
            link_p = [(t, r, d, s) for (t, r, d, s) in idx_z if dem[s][r][d] > 0]
            model.addConstrs((zp[t, r, d, s] <= y_p[t] for (t, r, d, s) in link_p), name='lp')
            model.addConstrs((zw[t, r, d, s] <= y_w[r2w[r], t] for (t, r, d, s) in link_p), name='lw')
            model.addConstrs((zr[t, r, d, s] <= y_r[r, t] for (t, r, d, s) in link_p), name='lr')

            # Sum = 1
            model.addConstrs((gp.quicksum(zp[t, r, d, s] for t in range(d + 1)) == 1
                              for s in range(S) for r in range(R) for d in range(T)), name='sp')
            model.addConstrs((gp.quicksum(zw[t, r, d, s] for t in range(d + 1)) == 1
                              for s in range(S) for r in range(R) for d in range(T)), name='sw')
            model.addConstrs((gp.quicksum(zr[t, r, d, s] for t in range(d + 1)) == 1
                              for s in range(S) for r in range(R) for d in range(T)), name='sr')

            # Inventory non-negativity: cumZp >= cumZw >= cumZr for t < d
            for s in range(S):
                for r in range(R):
                    for d in range(T):
                        if d == 0: continue
                        for t in range(d):
                            model.addConstr(
                                gp.quicksum(zp[tt, r, d, s] for tt in range(t + 1)) -
                                gp.quicksum(zw[tt, r, d, s] for tt in range(t + 1)) >= 0)
                            model.addConstr(
                                gp.quicksum(zw[tt, r, d, s] for tt in range(t + 1)) -
                                gp.quicksum(zr[tt, r, d, s] for tt in range(t + 1)) >= 0)

            # Objective via .Obj attributes (batch)
            coeffs = []; vars_ = []
            for t in range(T):
                coeffs.append(float(cp[t])); vars_.append(y_p[t])
            for w in range(W):
                for t in range(T):
                    coeffs.append(float(cw[w][t])); vars_.append(y_w[w, t])
            for r in range(R):
                for t in range(T):
                    coeffs.append(float(cr[r][t])); vars_.append(y_r[r, t])
            for (t, r, d, s) in idx_z:
                D = dem[s][r][d]
                if D == 0: continue
                dt = d - t
                p = probs[s]
                coeffs.append(p * D * hp * dt); vars_.append(zp[t, r, d, s])
                coeffs.append(p * D * (hw - hp) * dt); vars_.append(zw[t, r, d, s])
                coeffs.append(p * D * (hr[r] - hw) * dt); vars_.append(zr[t, r, d, s])
            obj_expr = gp.LinExpr()
            obj_expr.addTerms(coeffs, vars_)
            model.setObjective(obj_expr, GRB.MINIMIZE)

            # Warm start: all setups on, produce/ship at demand period
            for t in range(T): y_p[t].Start = 1
            for w in range(W):
                for t in range(T): y_w[w, t].Start = 1
            for r in range(R):
                for t in range(T): y_r[r, t].Start = 1
            for (t, r, d, s) in idx_z:
                v = 1.0 if t == d else 0.0
                zp[t, r, d, s].Start = v
                zw[t, r, d, s].Start = v
                zr[t, r, d, s].Start = v

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < best_holder[0] - 1e-9:
                        yp_v = model.cbGetSolution(y_p)
                        yw_v = model.cbGetSolution(y_w)
                        yr_v = model.cbGetSolution(y_r)
                        zp_v = model.cbGetSolution(zp)
                        zw_v = model.cbGetSolution(zw)
                        zr_v = model.cbGetSolution(zr)
                        P_v = {}; SW_v = {}; SR_v = {}; IP_v = {}; IW_v = {}; IR_v = {}
                        for s in range(S):
                            for t in range(T):
                                P_v[(t, s)] = 0.0; IP_v[(t, s)] = 0.0
                                for w in range(W):
                                    SW_v[(w, t, s)] = 0.0; IW_v[(w, t, s)] = 0.0
                                for r in range(R):
                                    SR_v[(r, t, s)] = 0.0; IR_v[(r, t, s)] = 0.0
                        for s in range(S):
                            for r in range(R):
                                for d in range(T):
                                    D = dem[s][r][d]
                                    w = r2w[r]
                                    czp = 0.0; czw = 0.0; czr = 0.0
                                    for t in range(d + 1):
                                        a = zp_v[t, r, d, s]; b = zw_v[t, r, d, s]; c = zr_v[t, r, d, s]
                                        P_v[(t, s)] += D * a
                                        SW_v[(w, t, s)] += D * b
                                        SR_v[(r, t, s)] += D * c
                                        czp += a; czw += b; czr += c
                                        if t < d:
                                            IP_v[(t, s)] += D * (czp - czw)
                                            IW_v[(w, t, s)] += D * (czw - czr)
                                            IR_v[(r, t, s)] += D * czr
                        sol = make_sol_from_aggregate(yp_v, yw_v, yr_v, P_v, SW_v, SR_v, IP_v, IW_v, IR_v, T, W, R, S)
                        best_holder[0] = obj; best_holder[1] = sol
                        if logger: logger.log_solution(obj, sol)

            model.optimize(callback=callback)

            if model.SolCount > 0 and best_holder[0] < best_obj - 1e-9:
                best_obj = best_holder[0]; best_sol = best_holder[1]

        else:
            # Aggregate big-M formulation
            P = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name='P')
            SW = model.addVars(W, T, S, vtype=GRB.CONTINUOUS, lb=0, name='SW')
            SR = model.addVars(R, T, S, vtype=GRB.CONTINUOUS, lb=0, name='SR')
            IP = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0, name='IP')
            IW = model.addVars(W, T, S, vtype=GRB.CONTINUOUS, lb=0, name='IW')
            IR = model.addVars(R, T, S, vtype=GRB.CONTINUOUS, lb=0, name='IR')

            Mp = {}; Mw = {}; Mr = {}
            for s in range(S):
                for r in range(R):
                    for t in range(T):
                        Mr[(r, t, s)] = sum(dem[s][r][d] for d in range(t, T))
                for w in range(W):
                    for t in range(T):
                        Mw[(w, t, s)] = sum(dem[s][r][d] for r in range(R) if r2w[r] == w for d in range(t, T))
                for t in range(T):
                    Mp[(t, s)] = sum(dem[s][r][d] for r in range(R) for d in range(t, T))

            for s in range(S):
                for t in range(T):
                    if t == 0:
                        model.addConstr(IP[t, s] == init_inv + P[t, s] - gp.quicksum(SW[w, t, s] for w in range(W)))
                    else:
                        model.addConstr(IP[t, s] == IP[t - 1, s] + P[t, s] - gp.quicksum(SW[w, t, s] for w in range(W)))
                    model.addConstr(P[t, s] <= Mp[(t, s)] * y_p[t])
                    for w in range(W):
                        if t == 0:
                            model.addConstr(IW[w, t, s] == init_inv + SW[w, t, s] -
                                            gp.quicksum(SR[r, t, s] for r in range(R) if r2w[r] == w))
                        else:
                            model.addConstr(IW[w, t, s] == IW[w, t - 1, s] + SW[w, t, s] -
                                            gp.quicksum(SR[r, t, s] for r in range(R) if r2w[r] == w))
                        model.addConstr(SW[w, t, s] <= Mw[(w, t, s)] * y_w[w, t])
                    for r in range(R):
                        if t == 0:
                            model.addConstr(IR[r, t, s] == init_inv + SR[r, t, s] - dem[s][r][t])
                        else:
                            model.addConstr(IR[r, t, s] == IR[r, t - 1, s] + SR[r, t, s] - dem[s][r][t])
                        model.addConstr(SR[r, t, s] <= Mr[(r, t, s)] * y_r[r, t])

            setup_cost = gp.quicksum(cp[t] * y_p[t] for t in range(T))
            for w in range(W):
                setup_cost += gp.quicksum(cw[w][t] * y_w[w, t] for t in range(T))
            for r in range(R):
                setup_cost += gp.quicksum(cr[r][t] * y_r[r, t] for t in range(T))
            holding = gp.LinExpr()
            for s in range(S):
                p = probs[s]
                for t in range(T):
                    holding += p * hp * IP[t, s]
                    for w in range(W):
                        holding += p * hw * IW[w, t, s]
                    for r in range(R):
                        holding += p * hr[r] * IR[r, t, s]
            model.setObjective(setup_cost + holding, GRB.MINIMIZE)

            # Warm start
            for t in range(T): y_p[t].Start = 1
            for w in range(W):
                for t in range(T): y_w[w, t].Start = 1
            for r in range(R):
                for t in range(T): y_r[r, t].Start = 1
            for s in range(S):
                for t in range(T):
                    td = sum(dem[s][r][t] for r in range(R))
                    P[t, s].Start = td
                    IP[t, s].Start = init_inv
                    for w in range(W):
                        wd = sum(dem[s][r][t] for r in range(R) if r2w[r] == w)
                        SW[w, t, s].Start = wd
                        IW[w, t, s].Start = init_inv
                    for r in range(R):
                        SR[r, t, s].Start = dem[s][r][t]
                        IR[r, t, s].Start = init_inv

            def callback_agg(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < best_holder[0] - 1e-9:
                        yp_v = model.cbGetSolution(y_p)
                        yw_v = model.cbGetSolution(y_w)
                        yr_v = model.cbGetSolution(y_r)
                        P_v = model.cbGetSolution(P)
                        SW_v = model.cbGetSolution(SW)
                        SR_v = model.cbGetSolution(SR)
                        IP_v = model.cbGetSolution(IP)
                        IW_v = model.cbGetSolution(IW)
                        IR_v = model.cbGetSolution(IR)
                        sol = make_sol_from_aggregate(yp_v, yw_v, yr_v, P_v, SW_v, SR_v, IP_v, IW_v, IR_v, T, W, R, S)
                        best_holder[0] = obj; best_holder[1] = sol
                        if logger: logger.log_solution(obj, sol)

            model.optimize(callback=callback_agg)

            if model.SolCount > 0 and best_holder[0] < best_obj - 1e-9:
                best_obj = best_holder[0]; best_sol = best_holder[1]

    except Exception as e:
        # On any Gurobi error, keep the fallback solution
        pass

    # Recompute objective from the solution for consistency
    final_obj = compute_objective(best_sol, inst)
    best_sol['objective_value'] = float(final_obj)

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f, indent=2)


if __name__ == '__main__':
    main()