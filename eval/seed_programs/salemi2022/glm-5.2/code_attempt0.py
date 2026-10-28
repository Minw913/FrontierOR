import json
import argparse
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n_gen = inst['n_generators']
    n_scen = inst['n_scenarios']
    T = inst['T']
    gens = inst['generators']
    demand = inst['demand_scenarios']
    reserve = inst['reserve_scenarios']

    M = [g['M'] for g in gens]
    m = [g['m'] for g in gens]
    L = [g['L'] for g in gens]
    ell = [g['ell'] for g in gens]
    RU = [g['RU'] for g in gens]
    SU = [g['SU'] for g in gens]
    RD = [g['RD'] for g in gens]
    SD = [g['SD'] for g in gens]
    c_g = [g['c_g'] for g in gens]
    c_f = [g['c_f'] for g in gens]
    K = [g['startup_costs_K'] for g in gens]

    # Precompute critical k values for start-up cost constraints
    critical_k = []
    for i in range(n_gen):
        Ki = K[i]
        ck = []
        prev = 0.0
        for k in range(1, T + 1):
            if Ki[k - 1] > prev:
                ck.append(k)
                prev = Ki[k - 1]
        critical_k.append(ck)

    model = gp.Model('unit_commitment')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = model.addVars(n_gen, T, vtype=GRB.BINARY, name='x')
    y = model.addVars(n_gen, T, vtype=GRB.BINARY, name='y')
    ybar = model.addVars(n_gen, T, vtype=GRB.BINARY, name='ybar')
    q = model.addVars(n_gen, T, lb=0, name='q')
    p = model.addVars(n_gen, n_scen, T, lb=0, name='p')
    pbar = model.addVars(n_gen, n_scen, T, lb=0, name='pbar')

    # Commitment logic: y - ybar = x - x_prev
    model.addConstrs(
        (y[i, 0] - ybar[i, 0] == x[i, 0] for i in range(n_gen)),
        name='logic_t0'
    )
    model.addConstrs(
        (y[i, t] - ybar[i, t] == x[i, t] - x[i, t - 1]
         for i in range(n_gen) for t in range(1, T)),
        name='logic'
    )

    # Minimum up-time
    model.addConstrs(
        (gp.quicksum(y[i, tau] for tau in range(max(0, t - L[i] + 1), t + 1)) <= x[i, t]
         for i in range(n_gen) for t in range(T)),
        name='min_up'
    )

    # Minimum down-time
    model.addConstrs(
        (gp.quicksum(ybar[i, tau] for tau in range(max(0, t - ell[i] + 1), t + 1)) <= 1 - x[i, t]
         for i in range(n_gen) for t in range(T)),
        name='min_down'
    )

    # Start-up cost constraints (only for critical k to reduce size)
    startup_constrs = []
    for i in range(n_gen):
        Ki = K[i]
        for t in range(T):
            for k in critical_k[i]:
                if k <= t:
                    start = max(0, t - k)
                    sum_x = gp.quicksum(x[i, tau] for tau in range(start, t))
                    startup_constrs.append(
                        q[i, t] >= Ki[k - 1] * (x[i, t] - sum_x)
                    )
    model.addConstrs(startup_constrs, name='startup_cost')

    # Production limits and availability
    model.addConstrs(
        (p[i, s, t] <= M[i] * x[i, t]
         for i in range(n_gen) for s in range(n_scen) for t in range(T)),
        name='p_max'
    )
    model.addConstrs(
        (p[i, s, t] >= m[i] * x[i, t]
         for i in range(n_gen) for s in range(n_scen) for t in range(T)),
        name='p_min'
    )
    model.addConstrs(
        (p[i, s, t] <= pbar[i, s, t]
         for i in range(n_gen) for s in range(n_scen) for t in range(T)),
        name='p_pbar'
    )
    model.addConstrs(
        (pbar[i, s, t] <= M[i] * x[i, t]
         for i in range(n_gen) for s in range(n_scen) for t in range(T)),
        name='pbar_max'
    )

    # Ramp-up
    model.addConstrs(
        (p[i, s, t] - p[i, s, t - 1] <= RU[i] * x[i, t - 1] + SU[i] * y[i, t]
         for i in range(n_gen) for s in range(n_scen) for t in range(1, T)),
        name='ramp_up'
    )

    # Ramp-down
    model.addConstrs(
        (p[i, s, t - 1] - p[i, s, t] <= RD[i] * x[i, t] + SD[i] * ybar[i, t]
         for i in range(n_gen) for s in range(n_scen) for t in range(1, T)),
        name='ramp_down'
    )

    # Demand satisfaction
    model.addConstrs(
        (gp.quicksum(p[i, s, t] for i in range(n_gen)) >= demand[s][t]
         for s in range(n_scen) for t in range(T)),
        name='demand'
    )

    # Reserve satisfaction
    model.addConstrs(
        (gp.quicksum(pbar[i, s, t] for i in range(n_gen)) >= demand[s][t] + reserve[s][t]
         for s in range(n_scen) for t in range(T)),
        name='reserve'
    )

    # Objective: expected cost
    fixed_cost = gp.quicksum(c_f[i] * x[i, t] for i in range(n_gen) for t in range(T))
    startup_cost = gp.quicksum(q[i, t] for i in range(n_gen) for t in range(T))
    prod_cost = gp.quicksum(
        c_g[i] * p[i, s, t]
        for i in range(n_gen) for s in range(n_scen) for t in range(T)
    ) / n_scen
    model.setObjective(fixed_cost + startup_cost + prod_cost, GRB.MINIMIZE)

    # Callback to log incumbent objectives
    class IncumbentLogger:
        def __init__(self, logger):
            self.logger = logger
            self.best_obj = float('inf')

        def __call__(self, model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < self.best_obj - 1e-6:
                    self.best_obj = obj
                    if self.logger:
                        self.logger.log(obj)

    callback = IncumbentLogger(logger)
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        x_vals = model.getAttr('X', x)
        y_vals = model.getAttr('X', y)
        ybar_vals = model.getAttr('X', ybar)
        q_vals = model.getAttr('X', q)
        p_vals = model.getAttr('X', p)
        pbar_vals = model.getAttr('X', pbar)

        x_list = [[int(round(x_vals[i, t])) for t in range(T)] for i in range(n_gen)]
        y_list = [[int(round(y_vals[i, t])) for t in range(T)] for i in range(n_gen)]
        ybar_list = [[int(round(ybar_vals[i, t])) for t in range(T)] for i in range(n_gen)]
        q_list = [[float(q_vals[i, t]) for t in range(T)] for i in range(n_gen)]
        p_list = [[[float(p_vals[i, s, t]) for t in range(T)] for s in range(n_scen)] for i in range(n_gen)]
        pbar_list = [[[float(pbar_vals[i, s, t]) for t in range(T)] for s in range(n_scen)] for i in range(n_gen)]

        objective_value = float(model.ObjVal)
    else:
        # Fallback: trivial solution (may not be feasible, but preserves schema)
        x_list = [[0] * T for _ in range(n_gen)]
        y_list = [[0] * T for _ in range(n_gen)]
        ybar_list = [[0] * T for _ in range(n_gen)]
        q_list = [[0.0] * T for _ in range(n_gen)]
        p_list = [[[0.0] * T for _ in range(n_scen)] for _ in range(n_gen)]
        pbar_list = [[[0.0] * T for _ in range(n_scen)] for _ in range(n_gen)]
        objective_value = float('inf')

    solution = {
        'objective_value': objective_value,
        'x': x_list,
        'y': y_list,
        'y_bar': ybar_list,
        'q': q_list,
        'p': p_list,
        'p_bar': pbar_list
    }

    if logger:
        logger.log_solution(objective_value, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()