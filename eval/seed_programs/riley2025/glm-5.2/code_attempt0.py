import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)

    nD = data['num_demand_sites']
    nP = data['num_production_sites']
    nS = data['num_scenarios']
    nT = data['num_time_periods']
    nM = data['num_module_sizes']
    U = data['units_per_module_size']
    u_max = data['u_max']
    probs = data['scenario_probabilities']
    cap = [ms['capacity'] for ms in data['module_sizes']]

    # Capital cost per (m, k)
    cap_cost = {}
    for mu in data['module_units']:
        cap_cost[(mu['module_size_id'], mu['unit_id'])] = mu['capital_cost']
    for m in range(nM):
        for k in range(U):
            if (m, k) not in cap_cost:
                cap_cost[(m, k)] = data['module_sizes'][m]['capital_cost_per_unit']

    pc = data['production_costs']           # [nD][nP]
    rc = data['relocation_costs']           # [nP][nP][nM]
    pen = data['penalty_costs']            # [nD][nT]
    scenarios = data['scenarios']
    dem = [scenarios[s]['demands'] for s in range(nS)]  # [s][i][t]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model('supply_chain')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, args.time_limit - 3)
    model.Params.OutputFlag = 0

    # ---- Variables ----
    y = model.addVars(nM, U, vtype=GRB.BINARY, name='y')
    x = model.addVars(nD, nP, nT, nS, lb=0.0, ub=1.0, name='x')
    q = model.addVars(nD, nT, nS, lb=0.0, ub=1.0, name='q')

    v = model.addVars(nP, nM, nT, nS, lb=0.0, vtype=GRB.INTEGER, name='v')
    vdepot = model.addVars(nM, nT, nS, lb=0.0, vtype=GRB.INTEGER, name='v_depot')
    wfd = model.addVars(nP, nM, nT, nS, lb=0.0, vtype=GRB.INTEGER, name='w_fd')
    wtd = model.addVars(nP, nM, nT, nS, lb=0.0, vtype=GRB.INTEGER, name='w_td')

    w_pairs = [(j, jp) for j in range(nP) for jp in range(nP) if j != jp]
    w = model.addVars(w_pairs, nM, nT, nS, lb=0.0, vtype=GRB.INTEGER, name='w')

    # z[m,s] expression = sum_k y[m,k]
    z_sum = {m: quicksum(y[m, k] for k in range(U)) for m in range(nM)}

    # ---- Demand fraction balance ----
    for s in range(nS):
        for t in range(nT):
            for i in range(nD):
                model.addConstr(
                    quicksum(x[i, j, t, s] for j in range(nP)) + q[i, t, s] == 1.0
                )

    # ---- Depot balance (per period) ----
    for s in range(nS):
        for t in range(nT):
            for m in range(nM):
                net = quicksum(wtd[j, m, t, s] - wfd[j, m, t, s] for j in range(nP))
                if t == 0:
                    model.addConstr(vdepot[m, t, s] == z_sum[m] + net)
                else:
                    model.addConstr(vdepot[m, t, s] == vdepot[m, t - 1, s] + net)

    # ---- Production site balance (per period) ----
    for s in range(nS):
        for t in range(nT):
            for j in range(nP):
                for m in range(nM):
                    inflow_pp = quicksum(w[jp, j, m, t, s] for jp in range(nP) if jp != j)
                    outflow_pp = quicksum(w[j, jp, m, t, s] for jp in range(nP) if jp != j)
                    net = inflow_pp + wfd[j, m, t, s] - outflow_pp - wtd[j, m, t, s]
                    if t == 0:
                        model.addConstr(v[j, m, t, s] == net)
                    else:
                        model.addConstr(v[j, m, t, s] == v[j, m, t - 1, s] + net)

    # ---- Capacity & global limit ----
    for s in range(nS):
        for t in range(nT):
            for j in range(nP):
                dw = quicksum(dem[s][i][t] * x[i, j, t, s] for i in range(nD))
                model.addConstr(dw <= quicksum(v[j, m, t, s] * cap[m] for m in range(nM)))
                model.addConstr(dw <= u_max)

    # ---- Objective ----
    capital = quicksum(cap_cost[(m, k)] * y[m, k] for m in range(nM) for k in range(U))

    operating = 0.0
    for s in range(nS):
        p = probs[s]
        cs = 0.0
        # serving + penalty
        for t in range(nT):
            for i in range(nD):
                for j in range(nP):
                    cs += pc[i][j] * dem[s][i][t] * x[i, j, t, s]
                cs += pen[i][t] * dem[s][i][t] * q[i, t, s]
        # relocation (only prod-to-prod; depot moves free)
        for t in range(nT):
            for m in range(nM):
                for j in range(nP):
                    for jp in range(nP):
                        if j != jp:
                            cs += rc[j][jp][m] * w[j, jp, m, t, s]
        operating += p * cs

    model.setObjective(capital + operating, GRB.MINIMIZE)

    # ---- Callback for incumbent logging ----
    best = [float('inf')]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best[0] - 1e-9:
                best[0] = obj
                if logger:
                    logger.log(obj)

    model.setCallback(cb)

    model.optimize()

    # ---- Fallback: if no integer solution, relax and resolve ----
    if model.SolCount == 0:
        # Relax integrality on second-stage integer vars, keep y binary
        for var in [v, vdepot, wfd, wtd, w]:
            for vv in var.values():
                vv.vtype = GRB.CONTINUOUS
        model.Params.TimeLimit = max(1, args.time_limit - 3)
        model.optimize()
        relaxed = True
    else:
        relaxed = False

    # ---- Build solution output ----
    patterns = {
        "y[m,k]": "m=0..num_module_sizes-1, k=0..units_per_module_size-1",
        "z[m,s]": "module size m, scenario s",
        "x[i,j,t,s]": "demand site i, production site j, period t, scenario s",
        "q[i,t,s]": "unmet-demand fraction",
        "v[j,m,t,s]": "units at production site",
        "v_depot[m,t,s]": "units at depot",
        "w[j,jp,m,t,s]": "relocation between distinct production sites j and jp",
        "w_fd[j,m,t,s]": "depot-to-production-site relocation",
        "w_td[j,m,t,s]": "production-site-to-depot relocation",
    }

    def build_solution():
        mv = {}
        ydict = {}
        if model.SolCount == 0:
            return mv, ydict
        for m in range(nM):
            zval = 0
            for k in range(U):
                val = y[m, k].X
                if val is not None and abs(val) > 1e-6:
                    iv = 1 if val > 0.5 else 0
                    mv[f'y[{m},{k}]'] = iv
                    ydict[f'y_{m}_{k}'] = iv
                    zval += iv
            if zval != 0:
                for s in range(nS):
                    mv[f'z[{m},{s}]'] = zval

        for s in range(nS):
            for t in range(nT):
                for i in range(nD):
                    val = q[i, t, s].X
                    if val is not None and abs(val) > 1e-6:
                        mv[f'q[{i},{t},{s}]'] = float(val)
                    for j in range(nP):
                        val = x[i, j, t, s].X
                        if val is not None and abs(val) > 1e-6:
                            mv[f'x[{i},{j},{t},{s}]'] = float(val)

        for s in range(nS):
            for t in range(nT):
                for m in range(nM):
                    val = vdepot[m, t, s].X
                    if val is not None and abs(val) > 1e-6:
                        mv[f'v_depot[{m},{t},{s}]'] = int(round(val))
                    for j in range(nP):
                        val = v[j, m, t, s].X
                        if val is not None and abs(val) > 1e-6:
                            mv[f'v[{j},{m},{t},{s}]'] = int(round(val))
                        val = wfd[j, m, t, s].X
                        if val is not None and abs(val) > 1e-6:
                            mv[f'w_fd[{j},{m},{t},{s}]'] = int(round(val))
                        val = wtd[j, m, t, s].X
                        if val is not None and abs(val) > 1e-6:
                            mv[f'w_td[{j},{m},{t},{s}]'] = int(round(val))
                        for jp in range(nP):
                            if j != jp:
                                val = w[j, jp, m, t, s].X
                                if val is not None and abs(val) > 1e-6:
                                    mv[f'w[{j},{jp},{m},{t},{s}]'] = int(round(val))
        return mv, ydict

    mv, ydict = build_solution()

    if model.SolCount > 0:
        obj_val = model.ObjVal
        if logger:
            logger.log_solution(obj_val, mv)
    else:
        obj_val = 0.0

    sol = {
        "objective_value": float(obj_val),
        "model_variables": mv,
        "y": ydict,
        "model_variable_name_patterns": patterns,
    }

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()