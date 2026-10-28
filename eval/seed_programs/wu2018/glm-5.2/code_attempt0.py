import json
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    N = inst['num_items']
    T = inst['num_periods']
    Mm = inst['num_machines']
    max_m_list = inst['max_machines_per_item_per_period']
    caps = inst['machine_capabilities']
    demand = inst['demand']
    holding = inst['holding_cost']
    capacity = inst['capacity']

    p_time_d = inst['production_time']
    s_cost_d = inst['setup_cost']
    s_time_d = inst['setup_time']
    p_cost_d = inst['production_cost']

    init_inv = inst.get('initial_inventory', None)
    if init_inv is None:
        init_inv = [0.0] * N

    # machine capabilities: convert 1-indexed machine ids to 0-indexed
    capable = [[m - 1 for m in caps[i]] for i in range(N)]

    # Detect indexing format of dict keys (item offset io, machine offset mo)
    def detect(d):
        best = (0, 1)
        best_cnt = -1
        for io_c in [0, 1]:
            for mo_c in [0, 1]:
                cnt = 0
                for i in range(N):
                    for m0 in capable[i]:
                        k = f'item_{i + io_c}_machine_{m0 + mo_c}'
                        if k in d:
                            cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = (io_c, mo_c)
        return best

    io, mo = detect(p_time_d)

    def gv(d, i, m0):
        k = f'item_{i + io}_machine_{m0 + mo}'
        return d[k]

    p_time, s_cost, s_time, p_cost = {}, {}, {}, {}
    for i in range(N):
        for m0 in capable[i]:
            p_time[(i, m0)] = gv(p_time_d, i, m0)
            s_cost[(i, m0)] = gv(s_cost_d, i, m0)
            s_time[(i, m0)] = gv(s_time_d, i, m0)
            p_cost[(i, m0)] = gv(p_cost_d, i, m0)

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('Threads', 1)
    env.setParam('Seed', 0)
    env.start()
    model = gp.Model(env=env)
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.TimeLimit = max(args.time_limit - 3, 1)

    X, Y, I = {}, {}, {}

    # Inventory variables
    for i in range(N):
        for t in range(T):
            I[i, t] = model.addVar(0.0, gp.GRB.INFINITY, float(holding[i]),
                                   name=f'I_{i}_{t}')

    # Setup and production variables
    for i in range(N):
        for m0 in capable[i]:
            for t in range(T):
                Y[i, m0, t] = model.addVar(0.0, 1.0, float(s_cost[(i, m0)]),
                                           vtype=GRB.BINARY,
                                           name=f'Y_{i}_{m0}_{t}')
                X[i, m0, t] = model.addVar(0.0, gp.GRB.INFINITY, float(p_cost[(i, m0)]),
                                           name=f'X_{i}_{m0}_{t}')

    model.update()

    # Flow conservation and max machines per item per period
    for i in range(N):
        for t in range(T):
            prev_inv = init_inv[i] if t == 0 else I[i, t - 1]
            model.addConstr(
                gp.quicksum(X[i, m0, t] for m0 in capable[i]) + prev_inv - I[i, t]
                == demand[i][t],
                name=f'flow_{i}_{t}'
            )
            model.addConstr(
                gp.quicksum(Y[i, m0, t] for m0 in capable[i]) <= max_m_list[i],
                name=f'maxm_{i}_{t}'
            )

    # Machine capacity constraints
    items_per_machine = [[] for _ in range(Mm)]
    for i in range(N):
        for m0 in capable[i]:
            items_per_machine[m0].append(i)

    for m0 in range(Mm):
        if not items_per_machine[m0]:
            continue
        for t in range(T):
            model.addConstr(
                gp.quicksum(p_time[(i, m0)] * X[i, m0, t] for i in items_per_machine[m0])
                + gp.quicksum(s_time[(i, m0)] * Y[i, m0, t] for i in items_per_machine[m0])
                <= capacity[m0][t],
                name=f'cap_{m0}_{t}'
            )

    # Linking (big-M) constraints
    for i in range(N):
        for m0 in capable[i]:
            for t in range(T):
                remaining = sum(demand[i][tp] for tp in range(t, T))
                cap_avail = capacity[m0][t] - s_time[(i, m0)]
                if p_time[(i, m0)] > 1e-9:
                    cap_units = cap_avail / p_time[(i, m0)]
                else:
                    cap_units = cap_avail
                bigM = min(remaining, cap_units)
                if bigM < 0:
                    bigM = 0.0
                if bigM > 1e-9:
                    model.addConstr(X[i, m0, t] <= bigM * Y[i, m0, t],
                                   name=f'link_{i}_{m0}_{t}')
                else:
                    model.addConstr(X[i, m0, t] == 0, name=f'link0_{i}_{m0}_{t}')
                    model.addConstr(Y[i, m0, t] == 0, name=f'forcen0_{i}_{m0}_{t}')

    model.update()

    # Helper to build solution dict from current solution values
    def build_sol_from_values(getter):
        prod, inv, setups = {}, {}, {}
        for i in range(N):
            for m0 in capable[i]:
                for t in range(T):
                    val = getter(X[i, m0, t])
                    prod[f'X_item{i + io}_machine{m0 + mo}_period{t}'] = float(val)
            for t in range(T):
                val = getter(I[i, t])
                inv[f'I_item{i + io}_period{t}'] = float(val)
            for m0 in capable[i]:
                for t in range(T):
                    val = getter(Y[i, m0, t])
                    setups[f'Y_item{i + io}_machine{m0 + mo}_period{t}'] = int(1 if val > 0.5 else 0)
        return {'production': prod, 'inventory': inv, 'setups': setups}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                sol = build_sol_from_values(lambda v: model.cbGetSolution(v))
                logger.log_solution(obj, sol)

    model.optimize(callback)

    if model.SolCount == 0:
        result = {
            'objective_value': 0.0,
            'production': {},
            'inventory': {},
            'setups': {}
        }
    else:
        obj = model.ObjVal
        sol = build_sol_from_values(lambda v: v.X)
        result = {
            'objective_value': float(obj),
            'production': sol['production'],
            'inventory': sol['inventory'],
            'setups': sol['setups']
        }
        if logger:
            logger.log_solution(float(obj), sol)

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()